// Native learned-feature frontend for ORB-SLAM3's Atlas / Map / g2o backend.
// No ORBextractor, Hamming matching, or binary vocabulary is called on this path.
#include "Atlas.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Optimizer.h"
#include "CameraModels/Pinhole.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <set>
#include <algorithm>
using namespace ORB_SLAM3;
struct Record { double time; Sophus::SE3f relative; KeyFrame* ref; bool valid; };
class MASt3RSystem {
    Atlas atlas{0};
    Pinhole* camera=nullptr;
    std::unordered_map<int,MapPoint*> points;
    std::vector<Record> trajectory;
    KeyFrame* lastKF=nullptr;
    int frameNumber=0, lastKFNumber=-100, baCalls=0;
    int relocalizations=0, loopClosures=0, globalBACalls=0, lastLoopFrame=-100;
    struct LoopRecord {int frame,query,reference,inliers,fused;};
    std::vector<LoopRecord> loops;
    cv::Mat Signature(const cv::Mat& d) {
        cv::Mat mean;cv::reduce(d,mean,0,cv::REDUCE_AVG,CV_32F);cv::normalize(mean,mean);return mean;
    }
    std::vector<MapPoint*> Match(Frame& f,KeyFrame* kf,bool projection) {
        cv::Mat bank;std::vector<MapPoint*> mp;
        for(int i=0;i<kf->N;i++){auto p=kf->GetMapPoint(i);if(p&&!p->isBad()){bank.push_back(kf->mDescriptors.row(i));mp.push_back(p);}}
        std::vector<MapPoint*> result(f.N,nullptr);if(bank.rows<2)return result;
        cv::BFMatcher matcher(cv::NORM_L2);std::vector<std::vector<cv::DMatch>> fw;std::vector<cv::DMatch> rev;
        matcher.knnMatch(f.mDescriptors,bank,fw,2);matcher.match(bank,f.mDescriptors,rev);
        std::set<MapPoint*> used;
        for(auto& row:fw){if(row.size()<2)continue;auto a=row[0],b=row[1];
            if(a.distance>=.85*b.distance || a.distance>.85 || rev[a.trainIdx].trainIdx!=a.queryIdx)continue;
            auto p=mp[a.trainIdx];if(used.count(p))continue;
            if(projection){auto x=f.GetPose()*p->GetWorldPos();if(x.z()<=0)continue;auto q=f.mpCamera->project(x);if((q-Eigen::Vector2f(f.mvKeysUn[a.queryIdx].pt.x,f.mvKeysUn[a.queryIdx].pt.y)).norm()>15)continue;}
            result[a.queryIdx]=p;used.insert(p);
        }return result;
    }
    int Verify(Frame& f,std::vector<MapPoint*>& matches,Sophus::SE3f& pose) {
        std::vector<cv::Point3f> obj;std::vector<cv::Point2f> pix;std::vector<int> ix;
        for(int i=0;i<f.N;i++)if(matches[i]&&!matches[i]->isBad()){auto x=matches[i]->GetWorldPos();obj.emplace_back(x.x(),x.y(),x.z());pix.push_back(f.mvKeysUn[i].pt);ix.push_back(i);}
        if(obj.size()<20)return 0;
        cv::Mat r,t,ins;bool ok=cv::solvePnPRansac(obj,pix,f.mK,cv::noArray(),r,t,false,200,3.,.999,ins,cv::SOLVEPNP_EPNP);
        if(!ok||ins.rows<20||ins.rows<float(obj.size())*.3)return 0;
        cv::Mat rr;cv::Rodrigues(r,rr);rr.convertTo(rr,CV_32F);t.convertTo(t,CV_32F);
        pose=Sophus::SE3f(Eigen::Quaternionf(Converter::toMatrix3f(rr)).normalized(),Eigen::Vector3f(t.at<float>(0),t.at<float>(1),t.at<float>(2)));
        std::vector<MapPoint*> good(f.N,nullptr);std::set<int> cells;
        for(int i=0;i<ins.rows;i++){int j=ix[ins.at<int>(i)];if((pose*matches[j]->GetWorldPos()).z()<=0)continue;good[j]=matches[j];auto uv=f.mvKeysUn[j].pt;cells.insert(int(uv.x/80)+4*int(uv.y/80));}
        if(cells.size()<4)return 0;matches.swap(good);return std::count_if(matches.begin(),matches.end(),[](MapPoint*p){return p!=nullptr;});
    }
public:
    void Track(const std::string& file) {
        auto start=std::chrono::steady_clock::now();
        cv::FileStorage in(file,cv::FileStorage::READ);
        if(!in.isOpened()) throw std::runtime_error("cannot read packet");
        cv::Mat K,uv,xyz,desc,ids,guess;
        in["K"]>>K; in["uv"]>>uv; in["xyz_world"]>>xyz; in["descriptors"]>>desc; in["ids"]>>ids; in["Tcw"]>>guess;
        // Strong CNN and compensated-geometry evidence may retire existing map points.
        cv::Mat rejected;in["rejected_ids"]>>rejected;int retired=0;
        for(int i=0;i<rejected.rows;i++) {
            auto it=points.find(rejected.at<int>(i,0));
            if(it!=points.end() && !it->second->isBad()){it->second->SetBadFlag();++retired;}
        }
        double timestamp=(double)in["timestamp"]; int width=(int)in["width"],height=(int)in["height"];
        bool forceRecovery=!in["force_relocalize"].empty() && int(in["force_relocalize"])!=0;
        if(uv.rows<6 || uv.cols!=2 || desc.rows!=uv.rows || desc.cols!=24 || xyz.rows!=uv.rows || xyz.cols!=3 || ids.rows!=uv.rows)
            throw std::runtime_error("invalid learned-feature packet");
        if(!cv::checkRange(uv)||!cv::checkRange(xyz)||!cv::checkRange(desc)||!cv::checkRange(guess)) throw std::runtime_error("nonfinite packet");
        if(!camera) {camera=new Pinhole({K.at<float>(0,0),K.at<float>(1,1),K.at<float>(0,2),K.at<float>(1,2)});atlas.AddCamera(camera);}
        Frame f;
        f.mnId=Frame::nNextId++;f.mTimeStamp=timestamp;f.N=uv.rows; f.Nleft=-1;f.Nright=-1;f.monoLeft=0;f.monoRight=0;
        f.mpCamera=camera;f.mpCamera2=nullptr;f.mpORBvocabulary=nullptr;f.mpORBextractorLeft=nullptr;f.mpORBextractorRight=nullptr;
        f.mpLastKeyFrame=lastKF;f.mpReferenceKF=lastKF;f.mnDataset=0;f.mnCloseMPs=0;
        f.mK=K;f.mK_=Converter::toMatrix3f(K);f.mDistCoef=cv::Mat::zeros(4,1,CV_32F);
        f.fx=K.at<float>(0,0);f.fy=K.at<float>(1,1);f.cx=K.at<float>(0,2);f.cy=K.at<float>(1,2);f.invfx=1/f.fx;f.invfy=1/f.fy;
        f.mbf=0;f.mb=0;f.mThDepth=0;f.mnScaleLevels=1;f.mfScaleFactor=1.2;f.mfLogScaleFactor=std::log(1.2);
        f.mvScaleFactors={1};f.mvInvScaleFactors={1};f.mvLevelSigma2={1};f.mvInvLevelSigma2={1};
        f.mnMinX=0;f.mnMinY=0;f.mnMaxX=width;f.mnMaxY=height;
        f.mfGridElementWidthInv=FRAME_GRID_COLS/float(width);f.mfGridElementHeightInv=FRAME_GRID_ROWS/float(height);
        f.mDescriptors=desc;f.mvuRight.assign(f.N,-1);f.mvDepth.assign(f.N,-1);f.mvbOutlier.assign(f.N,false);f.mvpMapPoints.assign(f.N,nullptr);
        std::vector<cv::Point3f> objects;std::vector<cv::Point2f> pixels;std::vector<int> indices;
        for(int i=0;i<f.N;i++) {
            cv::KeyPoint kp(uv.at<float>(i,0),uv.at<float>(i,1),8);kp.octave=0;f.mvKeys.push_back(kp);f.mvKeysUn.push_back(kp);
            int gx=std::min(FRAME_GRID_COLS-1,std::max(0,int(kp.pt.x*f.mfGridElementWidthInv)));
            int gy=std::min(FRAME_GRID_ROWS-1,std::max(0,int(kp.pt.y*f.mfGridElementHeightInv)));f.mGrid[gx][gy].push_back(i);
            auto it=points.find(ids.at<int>(i,0));
            if(it!=points.end() && it->second->isBad() && it->second->GetReplaced())it->second=it->second->GetReplaced();
            if(it!=points.end() && !it->second->isBad()) {
                auto p=it->second;auto x=p->GetWorldPos();f.mvpMapPoints[i]=p;
                objects.emplace_back(x.x(),x.y(),x.z());pixels.push_back(kp.pt);indices.push_back(i);
            }
        }
        Eigen::Matrix3f R;Eigen::Vector3f t;
        for(int r=0;r<3;r++){for(int c=0;c<3;c++)R(r,c)=guess.at<float>(r,c);t(r)=guess.at<float>(r,3);}
        f.SetPose(Sophus::SE3f(Eigen::Quaternionf(R).normalized(),t));
        int localAdded=0;
        auto allKFs=atlas.GetAllKeyFrames();std::sort(allKFs.begin(),allKFs.end(),[](KeyFrame*a,KeyFrame*b){return a->mnId>b->mnId;});
        if(lastKF&&!forceRecovery){int seen=0;for(auto k:allKFs){if(k->isBad())continue;auto matches=Match(f,k,true);for(int i=0;i<f.N;i++)if(!f.mvpMapPoints[i]&&matches[i]){f.mvpMapPoints[i]=matches[i];++localAdded;}if(++seen==5)break;}}
        if(forceRecovery)std::fill(f.mvpMapPoints.begin(),f.mvpMapPoints.end(),nullptr);
        objects.clear();pixels.clear();indices.clear();
        for(int i=0;i<f.N;i++)if(f.mvpMapPoints[i]){auto x=f.mvpMapPoints[i]->GetWorldPos();objects.emplace_back(x.x(),x.y(),x.z());pixels.push_back(f.mvKeysUn[i].pt);indices.push_back(i);}
        bool initialized=lastKF!=nullptr;int inliers=0;bool valid=!initialized;
        if(initialized && objects.size()>=12) {
            cv::Mat rv,tv,idx;
            bool ok=cv::solvePnPRansac(objects,pixels,K,cv::noArray(),rv,tv,false,150,3.0,0.999,idx,cv::SOLVEPNP_EPNP);
            if(ok && idx.rows>=12) {
                cv::Mat rr;cv::Rodrigues(rv,rr);rr.convertTo(rr,CV_32F);tv.convertTo(tv,CV_32F);
                R=Converter::toMatrix3f(rr);t=Eigen::Vector3f(tv.at<float>(0),tv.at<float>(1),tv.at<float>(2));
                f.SetPose(Sophus::SE3f(Eigen::Quaternionf(R).normalized(),t));
                inliers=Optimizer::PoseOptimization(&f);valid=inliers>=12;
            }
        }
        KeyFrame* loopKF=nullptr;int loopInliers=0,fused=0;bool recovered=false;
        std::vector<MapPoint*> historyMatches;Sophus::SE3f historyPose;
        if(initialized && (!valid || (frameNumber%5==0 && frameNumber-lastLoopFrame>=30))) {
            std::vector<std::pair<float,KeyFrame*>> ranked;auto signature=Signature(desc);
            for(auto k:allKFs){if(k->isBad()||(valid&&frameNumber-int(k->mnFrameId)<30))continue;ranked.emplace_back(float(cv::norm(signature,Signature(k->mDescriptors))),k);}
            std::sort(ranked.begin(),ranked.end(),[](const auto&a,const auto&b){return a.first<b.first;});
            for(size_t q=0;q<std::min(size_t(5),ranked.size());q++){auto k=ranked[q].second;auto matches=Match(f,k,false);Sophus::SE3f pose;int n=Verify(f,matches,pose);if(n>loopInliers){loopInliers=n;loopKF=k;historyMatches=matches;historyPose=pose;}}
            if(loopKF && loopInliers>=30){
                if(!valid){f.mvpMapPoints=historyMatches;f.SetPose(historyPose);inliers=Optimizer::PoseOptimization(&f);valid=inliers>=20;recovered=valid;if(valid)++relocalizations;loopKF=nullptr;}
                else {
                    // Candidate must agree with the locally tracked pose before adding closure observations.
                    float shift=(historyPose.inverse().translation()-f.GetPose().inverse().translation()).norm();
                    if(shift>.75f)loopKF=nullptr;
                    else {for(int i=0;i<f.N;i++)if(historyMatches[i]){auto old=f.mvpMapPoints[i];auto p=historyMatches[i];if(old&&old!=p&&!old->isBad()&&(old->GetWorldPos()-p->GetWorldPos()).norm()<.10f){old->Replace(p);++fused;}f.mvpMapPoints[i]=p;}f.SetPose(historyPose);inliers=Optimizer::PoseOptimization(&f);valid=inliers>=20;}
                }
            }else loopKF=nullptr;
        }
        // Lost frames cannot insert points; never silently substitute a guessed pose.
        bool keyframe=valid && (!initialized || frameNumber-lastKFNumber>=5 || inliers<100 || loopKF || recovered);
        for(int i=0;i<f.N;i++) if(f.mvbOutlier[i] || !valid) f.mvpMapPoints[i]=nullptr;
        int fixed=0,opt=0,nmp=0,edges=0;
        if(keyframe) {
            auto map=atlas.GetCurrentMap();auto kf=new KeyFrame(f,map,nullptr);atlas.AddKeyFrame(kf);
            if(!initialized) {map->mvpKeyFrameOrigins.push_back(kf);}
            for(int i=0;i<f.N;i++) {
                auto p=f.mvpMapPoints[i];
                if(!p) {
                    // Do not reincarnate a rejected known correspondence as a new map point.
                    if(ids.at<int>(i,0)>=0) continue;
                    Eigen::Vector3f x(xyz.at<float>(i,0),xyz.at<float>(i,1),xyz.at<float>(i,2));
                    if((f.GetPose()*x).z()<=0) continue;
                    p=new MapPoint(x,kf,map);atlas.AddMapPoint(p);points[int(p->mnId)]=p;f.mvpMapPoints[i]=p;
                }
                p->AddObservation(kf,i);kf->AddMapPoint(p,i);p->UpdateNormalAndDepth();
            }
            kf->UpdateConnections();lastKF=kf;lastKFNumber=frameNumber;
            if(atlas.GetAllKeyFrames().size()>=3 && atlas.GetAllKeyFrames().size()%2==1) {
                bool stop=false;Optimizer::LocalBundleAdjustment(kf,&stop,map,fixed,opt,nmp,edges);++baCalls;
                f.SetPose(kf->GetPose());
            }
            if(loopKF){kf->AddLoopEdge(loopKF);loopKF->AddLoopEdge(kf);kf->UpdateConnections();loopKF->UpdateConnections();Optimizer::GlobalBundleAdjustemnt(map,5,nullptr,0,true);f.SetPose(kf->GetPose());++globalBACalls;++loopClosures;lastLoopFrame=frameNumber;loops.push_back({frameNumber,int(kf->mnId),int(loopKF->mnId),loopInliers,fused});}
        }
        if(valid && lastKF) trajectory.push_back({timestamp,f.GetPose()*lastKF->GetPoseInverse(),lastKF,true});
        cv::Mat outIds(f.N,1,CV_32S,cv::Scalar(-1)),world=cv::Mat::zeros(f.N,3,CV_32F);
        for(int i=0;i<f.N;i++) if(f.mvpMapPoints[i] && !f.mvpMapPoints[i]->isBad()) {
            auto p=f.mvpMapPoints[i];outIds.at<int>(i)=int(p->mnId);auto x=p->GetWorldPos();for(int c=0;c<3;c++)world.at<float>(i,c)=x(c);
        }
        cv::Mat pose=Converter::toCvMat(f.GetPose().matrix());
        cv::FileStorage out(file+".result.yml",cv::FileStorage::WRITE);
        out<<"valid"<<int(valid)<<"keyframe"<<int(keyframe)<<"keyframe_id"<<(lastKF?int(lastKF->mnId):-1)<<"inliers"<<inliers<<"retired_map_points"<<retired;
        out<<"associated"<<int(objects.size())<<"ids"<<outIds<<"world"<<world<<"Tcw"<<pose;
        out<<"local_added"<<localAdded<<"relocalized"<<int(recovered)<<"relocalizations"<<relocalizations<<"loop_closures"<<loopClosures<<"loop_inliers"<<loopInliers<<"global_ba_calls"<<globalBACalls<<"fused_map_points"<<fused;
        out<<"map_points"<<int(atlas.GetAllMapPoints().size())<<"keyframes"<<int(atlas.GetAllKeyFrames().size());
        out<<"ba_calls"<<baCalls<<"ba_edges"<<edges<<"backend_ms"<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        out.release();++frameNumber;
    }
    void Save(const std::string& dir) {
        std::ofstream events(dir+"/loop_events.csv");events<<"frame,query_kf,reference_kf,inliers,fused\n";for(auto e:loops)events<<e.frame<<","<<e.query<<","<<e.reference<<","<<e.inliers<<","<<e.fused<<"\n";
        std::ofstream traj(dir+"/trajectory.txt");traj<<std::fixed<<std::setprecision(9);
        for(auto& rec:trajectory) {auto Twc=(rec.relative*rec.ref->GetPose()).inverse();auto t=Twc.translation();auto q=Twc.unit_quaternion();traj<<rec.time<<" "<<t.x()<<" "<<t.y()<<" "<<t.z()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";}
        std::vector<MapPoint*> good;for(auto p:atlas.GetAllMapPoints())if(!p->isBad())good.push_back(p);
        std::ofstream ply(dir+"/sparse_map.ply");ply<<"ply\nformat ascii 1.0\nelement vertex "<<good.size()<<"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        for(auto p:good){auto x=p->GetWorldPos();ply<<x.x()<<" "<<x.y()<<" "<<x.z()<<"\n";}
        cv::FileStorage kfs(dir+"/keyframes.yml",cv::FileStorage::WRITE);kfs<<"keyframes"<<"[";
        for(auto k:atlas.GetAllKeyFrames())if(!k->isBad())kfs<<"{"<<"id"<<int(k->mnId)<<"timestamp"<<k->mTimeStamp<<"Tcw"<<Converter::toCvMat(k->GetPose().matrix())<<"}";
        kfs<<"]";kfs.release();
    }
};
int main(){
    cv::setNumThreads(2);MASt3RSystem system;std::string op,path;
    while(std::cin>>op>>path){try{if(op=="TRACK")system.Track(path);else if(op=="SAVE")system.Save(path);else if(op=="QUIT")break;else throw std::runtime_error("bad command");std::cout<<"DONE "<<path<<std::endl;}catch(const std::exception& e){std::cout<<"ERROR "<<e.what()<<std::endl;return 2;}}
    return 0;
}

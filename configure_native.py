from pathlib import Path
import shutil, difflib
root=Path(__file__).resolve().parent
src=root/'third_party/ORB_SLAM3'
p=src/'CMakeLists.txt'
pristine=root/'pristine/CMakeLists.txt';pristine.parent.mkdir(exist_ok=True)
if not pristine.exists():shutil.copy2(p,pristine)
text=pristine.read_text().split('# Build examples')[0].replace('-std=c++11','-std=c++14')
text+='\nadd_executable(mast3r_backend ../../native/mast3r_backend.cc)\ntarget_link_libraries(mast3r_backend ${PROJECT_NAME})\n'
p.write_text(text)
matcher=src/'src/ORBmatcher.cc'
mt=matcher.read_text()
if 'Learned MapPoint descriptor medoid' not in mt:
    needle='int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)\n    {'
    assert needle in mt
    mt=mt.replace(needle,needle+'\n        // Learned MapPoint descriptor medoid during learned-point fusion.\n        if(a.type()==CV_32F && b.type()==CV_32F) return cvRound(1000*cv::norm(a,b,cv::NORM_L2));',1)
    matcher.write_text(mt)
(root/'orbslam3_integration.patch').write_text(''.join(difflib.unified_diff(pristine.read_text().splitlines(True),text.splitlines(True),fromfile='a/CMakeLists.txt',tofile='b/CMakeLists.txt')))
print('Configured independent native ORB-SLAM3 backend')

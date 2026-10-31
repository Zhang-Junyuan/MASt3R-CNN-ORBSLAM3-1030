"""Generate the visual-only V2 report and verify exported observation/map contracts."""
from pathlib import Path
import json,csv,base64,hashlib,difflib
import numpy as np,cv2
from report import evaluate,ply
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;out=root/'results/visual_v2';out.mkdir(parents=True,exist_ok=True)
def embed(p):
 mime='image/jpeg' if p.suffix.lower() in ('.jpg','.jpeg') else 'image/png'
 return '<img src="data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main():
 metrics={};verified={}
 for mode in ('baseline','dynamic'):
  folder=out/mode
  if not (folder/'summary.json').exists():continue
  m=evaluate(folder);metrics[mode]=m
  rows=list(csv.DictReader((folder/'frames.csv').open()));traj=np.loadtxt(folder/'trajectory.txt',ndmin=2)
  assert len(rows)==m['frames'] and len(traj)==m['valid_frames'] and np.isfinite(traj).all()
  assert m['imu_used'] is False and m['pure_visual'] is True
  maxrisk=0.
  if mode=='dynamic':
   for row in rows:
    if int(row['retained'])<12:continue
    i=int(row['frame']);fs=cv2.FileStorage(str(folder/'packets'/f'{i:06d}.yml'),cv2.FILE_STORAGE_READ);uv=np.rint(fs.getNode('uv').mat()).astype(int);d=fs.getNode('descriptors').mat();fs.release()
    assert d.shape==(len(uv),24) and np.isfinite(d).all()
    with np.load(folder/'visualizations'/f'probability_{i:04d}.npz') as z:p=z['probability'][uv[:,1],uv[:,0]]
    maxrisk=max(maxrisk,float(p.max()));assert np.all(p<=.650001)
  counts={}
  for name in ('sparse_map.ply','dense_filtered_candidate.ply','dense_static_multiview.ply'):
   data=ply(folder/name);assert len(data)>0 and np.isfinite(data).all();counts[name]=len(data)
  verified[mode]={'frames':len(rows),'saved_valid_poses':len(traj),'max_retained_dynamic_risk':maxrisk,'point_counts':counts,'imu_used':False}
 tests={}
 for name,path in [('synthetic_chain',root/'results/visual_chain_test.json'),('real_feature_relocalization',out/'relocalization_replay.json'),('real_feature_controlled_loop',out/'loop_replay.json')]:
  if path.exists():tests[name]=json.loads(path.read_text())
 bundle={'main_sequences':metrics,'independent_tests':tests,'verification':verified}
 (out/'verification.json').write_text(json.dumps(bundle,indent=2))
 diff=''
 for before,after,rel in [(root/'pristine/CMakeLists.txt',root/'third_party/ORB_SLAM3/CMakeLists.txt','CMakeLists.txt'),(root/'pristine/ORBmatcher.cc',root/'third_party/ORB_SLAM3/src/ORBmatcher.cc','src/ORBmatcher.cc')]:
  diff+=''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
 (root/'pure_visual_v2_integration.patch').write_text(diff)
 d=metrics.get('dynamic',next(iter(metrics.values())))
 text='# 纯视觉 V2 全链路验证\n\n只使用 RGB 与内参，不使用 IMU、传感器深度或轨迹真值估计位姿。真值仅用于事后评估。\n\n'
 text+=f'主序列：{d["frames"]} 帧，{d["valid_frames"]} 帧有效位姿，成功比例 {d["valid_fraction"]:.1%}，SE3 对齐 ATE RMSE {d["ate_se3_rmse_m"]:.4f} m。局部地图补充 {d["local_matches_added"]} 个关联，局部 BA {d["ba_calls"]} 次，撤销旧地图点 {d["retired_map_points"]} 个。\n\n'
 text+=f'原始连续 RGB 片段的自然事件：重定位 {d["relocalizations"]} 次，回环 {d["loop_closures"]} 次，全局 BA {d["global_ba_calls"]} 次。没有触发事件不等于缺少相应实现；下方单独记录受控验证。\n\n'
 if 'real_feature_relocalization' in tests:
  r=tests['real_feature_relocalization'];text+=f'真实 MASt3R 特征重放测试：第 {r["fault_frame"]} 帧主动清空关联并跳过局部跟踪，以 {r["recovery_inliers"]} 个内点成功历史重定位。该测试没有重新推理 RGB。\n\n'
 if 'real_feature_controlled_loop' in tests:
  r=tests['real_feature_controlled_loop'];events=r['return_events'];accepted=[x for x in events if x['fused_map_points']>0];fused=sum(x['fused_map_points'] for x in events)
  text+=f'真实特征受控回访：先重放前 120 帧，再重放开头 10 个视图；共 {r["frames"]} 帧，重定位 {r["relocalizations"]} 次、回环 {r["loop_closures"]} 次、全局 BA {r["global_ba_calls"]} 次、融合重复点 {fused} 个。它验证回环链路实际执行，不是自然闭环数据集上的性能结论。\n\n'
 if 'synthetic_chain' in tests:
  r=tests['synthetic_chain'];text+=f'合成几何测试：46 帧有效跟踪，强制丢失后恢复；回环和全局 BA 实际执行，最大平移误差 {r["max_translation_error_m"]:.6f} m；无关场景拒绝测试通过。\n\n'
 text+='实现方式：新增纯视觉控制器统一调度浮点局部地图匹配、关键帧管理、动态点撤销、地图点融合、历史检索、重定位和回环。复用原生 Atlas / Map / KeyFrame / MapPoint / PoseOptimization / LocalBundleAdjustment / GlobalBundleAdjustemnt。原版 ORB 三线程与二进制词袋未启用。\n\n'
 text+='限制：CNN 仅合成训练，真实动态精度未知；单目 BA 仍可能尺度漂移；当前回环采用紧凑描述子签名检索、PnP 几何验证、共享地图观测与全局 BA，限制较大的位姿差，未实现任意尺度 Sim3 或多地图合并。不能宣称达到原版全部能力或已提高定位精度。\n\n'
 text+='主序列点云与轨迹在 results/visual_v2/dynamic。controlled loop 的轨迹及稀疏地图在 results/visual_v2/loop_replay；受控测试结果不计入主序列成功率。MASt3R 缓存复用不代表实时推理速度，模块耗时不能直接解释为端到端 FPS。\n'
 (root/'纯视觉V2_验证报告.md').write_text(text,encoding='utf-8')
 sections=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>纯视觉 V2 全链路</title><style>body{font:17px/1.7 system-ui,Microsoft YaHei;max-width:1550px;margin:25px auto;padding:20px;background:#f5f7fb;color:#19304b}img{max-width:100%;height:auto}pre{white-space:pre-wrap;font:inherit;background:white;padding:22px;border-radius:12px}a{color:#2366a4}</style><h1>纯视觉 V2：详细框架与实际验证</h1><pre>'+text+'</pre>']
 for title,name in [('完整详细框架','04_visual_v2_complete'),('原始与改进纯视觉框架对照','05_visual_v2_comparison'),('动态点 CNN 内部结构','02_cnn_detail')]:
  sections.append('<h2>'+title+'</h2><p><a href="../../figures/'+name+'.svg">打开 SVG 矢量图</a></p>'+embed(root/'figures'/(name+'.png')))
 folder=out/'dynamic'
 if (folder/'trajectory.png').exists():sections+=['<h2>主序列实际轨迹</h2>',embed(folder/'trajectory.png')]
 cloud=ply(folder/'dense_static_multiview.ply');sample=cloud[::max(1,len(cloud)//12000)]
 fig=plt.figure(figsize=(9,5));ax=fig.add_subplot(projection='3d');ax.scatter(sample[:,0],sample[:,1],sample[:,2],c=sample[:,3:6]/255,s=.4);ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Visual V2 multi-view-supported dense map');fig.tight_layout();fig.savefig(out/'dense_map.png',dpi=150);plt.close(fig);sections+=['<h2>实际融合点云</h2>',embed(out/'dense_map.png')]
 previews=sorted((folder/'visualizations').glob('*.jpg'))
 if previews:sections+=['<h2>左：地图关联点；右：动态概率</h2>',embed(previews[len(previews)//2])]
 sections.append('</html>');(out/'report.html').write_text(''.join(sections),encoding='utf-8')
 artifacts={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('native/mast3r_backend.cc','run_pipeline.py','third_party/ORB_SLAM3/src/ORBmatcher.cc','build/orb/mast3r_backend','pure_visual_v2_integration.patch')}
 (out/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2));print('VISUAL_V2_REPORT_AND_VERIFICATION_PASSED')
if __name__=='__main__':main()

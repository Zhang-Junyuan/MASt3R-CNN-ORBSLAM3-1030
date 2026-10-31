from pathlib import Path
import json,csv,base64,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def ply(path):
    with open(path) as f:
        while f.readline().strip()!='end_header':pass
        rows=[list(map(float,l.split())) for l in f if l.strip()]
    return np.array(rows,dtype=np.float32)
def align(pred,gt):
    a=pred-pred.mean(0);b=gt-gt.mean(0);u,_,vt=np.linalg.svd(a.T@b);R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    return a@R.T+gt.mean(0)
def evaluate(folder):
    summary=json.loads((folder/'summary.json').read_text());frames=list(csv.DictReader((folder/'frames.csv').open()))
    metrics={**summary}
    metrics['retired_map_points_total']=sum(int(r.get('retired_map_points',0)) for r in frames)
    for key in ('mast3r_ms','gate_ms','backend_ms','frame_wall_ms'):
        values=np.array([float(r[key]) for r in frames]);metrics[key+'_mean']=float(values.mean());metrics[key+'_p95']=float(np.percentile(values,95))
    metrics['cached_frames']=sum(int(r['cache_hit']) for r in frames)
    traj=np.loadtxt(folder/'trajectory.txt',ndmin=2);gt=np.loadtxt(Path(summary['dataset'])/'groundtruth.txt')
    idx=np.abs(traj[:,0,None]-gt[None,:,0]).argmin(1);good=np.abs(gt[idx,0]-traj[:,0])<.025
    if good.sum()>=3:
        truth=gt[idx[good],1:4];pred=align(traj[good,1:4],truth);err=np.linalg.norm(pred-truth,axis=1)
        metrics['ate_se3_rmse_m']=float(np.sqrt(np.mean(err**2)));metrics['ate_associated_poses']=int(good.sum())
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(truth[:,0],truth[:,2],'k-',label='Ground truth');ax.plot(pred[:,0],pred[:,2],label=summary['mode']);ax.set(xlabel='x (m)',ylabel='z (m)',title='SE(3) aligned trajectory (no scale fitting)');ax.axis('equal');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(folder/'trajectory.png',dpi=150);plt.close(fig)
    else:metrics['ate_se3_rmse_m']=None
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
def b64image(path):
    mime='image/jpeg' if path.suffix.lower() in ('.jpg','.jpeg') else 'image/png'
    return '<img style="max-width:100%" src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def architecture():
    labels=[('双帧 RGB','标定与尺寸同步'),('MASt3R','点图 + 24 维描述子 + 置信度'),('相机运动补偿','鲁棒初始位姿 / 重投影残差'),('卷积动态过滤','有序 XYZ / RGB / 残差 → Pdyn'),('稀疏化与关联','网格选点 + 体素 + 浮点匹配'),('ORB-SLAM3 后端','PnP + 位姿优化 + 局部 BA')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="330" viewBox="0 0 1680 330"><rect width="1680" height="330" fill="#f5f8fc"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#526a87"/></marker></defs><g font-family="Microsoft YaHei, sans-serif"><text x="30" y="38" font-size="23" fill="#172d4c">MASt3R 动态过滤与稀疏地图：当前已实现路径</text>']
    for i,(a,b) in enumerate(labels):
        x=24+i*276;svg.append(f'<rect x="{x}" y="75" width="250" height="100" rx="12" fill="white" stroke="#477cb3"/><text x="{x+125}" y="111" text-anchor="middle" font-size="20">{a}</text><text x="{x+125}" y="144" text-anchor="middle" font-size="14">{b}</text>')
        if i<5:svg.append(f'<path d="M{x+250} 125h23" stroke="#526a87" stroke-width="2" marker-end="url(#a)"/>')
    svg.append('<text x="28" y="222" font-size="19">关键帧静态候选稠密点 → 用优化后的关键帧位姿重投影到世界 → 体素融合 → 多视角支持点云</text><text x="28" y="261" font-size="16" fill="#8e4527">边界：卷积模型仅合成训练；不含已适配的词袋回环、IMU 或语义分割；未宣称实时或动态检测精度。</text><text x="28" y="294" font-size="16" fill="#526a87">保留 Atlas / Map / KeyFrame / MapPoint / Optimizer；新增前端直接调用原生 C++ 后端，不调用 ORB 特征提取。</text></g></svg>')
    (ROOT/'architecture.svg').write_text(''.join(svg),encoding='utf-8')
def main():
    architecture();metrics=[];sections=[]
    for name in ('baseline','dynamic'):
        folder=OUT/name
        if not (folder/'summary.json').exists():continue
        m=evaluate(folder);metrics.append(m)
        sections.append(f'<h2>{name}</h2><p>处理 {m["frames"]} 帧；有效位姿 {m["valid_frames"]} 帧；ATE {m["ate_se3_rmse_m"]:.4f} m；关键帧 {m["keyframes"]}；局部 BA {m["ba_calls"]} 次。</p>')
        if (folder/'trajectory.png').exists():sections.append(b64image(folder/'trajectory.png'))
        previews=sorted((folder/'visualizations').glob('*.jpg'))
        if previews:sections.append(b64image(previews[len(previews)//2]))
    (OUT/'comparison.json').write_text(json.dumps(metrics,indent=2))
    table='<table><tr><th>配置</th><th>有效位姿</th><th>ATE m</th><th>MASt3R ms</th><th>CNN ms</th><th>后端 ms</th><th>CNN 剔除数</th><th>多视角稠密点</th></tr>'
    for m in metrics:table+=f'<tr><td>{m["mode"]}</td><td>{m["valid_fraction"]:.1%}</td><td>{m["ate_se3_rmse_m"]:.4f}</td><td>{m["mast3r_ms_mean"]:.1f}</td><td>{m["gate_ms_mean"]:.2f}</td><td>{m["backend_ms_mean"]:.1f}</td><td>{m["cnn_rejected_total"]}</td><td>{m["dense"]["multiview_voxels"]}</td></tr>'
    table+='</table>'
    points=[]
    cloud=OUT/'dynamic/dense_static_multiview.ply'
    if cloud.exists():
        data=ply(cloud)
        if len(data):
            data=data[::max(1,len(data)//10000)];points=data.tolist()
            fig=plt.figure(figsize=(8,5));ax=fig.add_subplot(projection='3d');ax.scatter(data[:,0],data[:,1],data[:,2],c=data[:,3:6]/255,s=.4);ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Filtered multi-view dense map');fig.tight_layout();fig.savefig(OUT/'dense_map.png',dpi=160);plt.close(fig)
    viewer='''<h2>实际点云预览</h2><p>拖动旋转，滚轮缩放。显示过滤后具有至少两个关键帧支持的点云抽样。</p><canvas id="cloud" width="1100" height="520"></canvas><script>
const pts=POINTS;const c=document.getElementById('cloud'),ctx=c.getContext('2d');let yaw=.2,pitch=.1,zoom=130,down=false,last;
let center=[0,0,0];if(pts.length){for(const p of pts)for(let k=0;k<3;k++)center[k]+=p[k]/pts.length;}
function draw(){ctx.fillStyle='#0c1523';ctx.fillRect(0,0,c.width,c.height);let q=[];for(const p of pts){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];let a=x*Math.cos(yaw)+z*Math.sin(yaw),b=-x*Math.sin(yaw)+z*Math.cos(yaw),d=y*Math.cos(pitch)-b*Math.sin(pitch),e=y*Math.sin(pitch)+b*Math.cos(pitch);q.push([a,d,e,p]);}q.sort((a,b)=>b[2]-a[2]);for(const [x,y,z,p] of q){ctx.fillStyle=`rgb(${p[3]},${p[4]},${p[5]})`;ctx.fillRect(c.width/2+x*zoom,c.height/2+y*zoom,2,2);}}
c.onpointerdown=e=>{down=true;last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>down=false;c.onpointermove=e=>{if(!down)return;yaw+=(e.clientX-last[0])*.008;pitch+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY];draw()};c.onwheel=e=>{e.preventDefault();zoom*=e.deltaY>0?.9:1.1;draw()};draw();</script>'''.replace('POINTS',json.dumps(points))
    limits='本项目已实现 RGB → MASt3R 点图 → 卷积过滤 → 稀疏关键点 → 原生 ORB-SLAM3 地图与优化链路。baseline 是关闭过滤的同一 MASt3R 系统，不是原始 ORB-SLAM3。CNN 只用合成残差训练，真实动态检测准确率未知；剔除数不等于正确动态点数。当前不含适配后的词袋回环、IMU、全局重定位或实时保证。短序列一次运行的 ATE 只能验证链路，不能证明创新有效。'
    timing='MASt3R 时间为实际首次推理所记录的耗时，缓存复用不会算作加速结果。后端包括 YAML 读取、PnP、位姿优化与部分帧的局部 BA；CNN 时间含 GPU 传输。各模块计时不是端到端 FPS。'
    body='<!doctype html><html lang="zh"><meta charset="utf-8"><title>MASt3R 动态 ORB-SLAM3 验证</title><style>body{font:16px/1.7 system-ui,Microsoft YaHei;background:#f4f7fb;color:#192b44;max-width:1160px;margin:35px auto;padding:20px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #d3dce8}canvas{max-width:100%;touch-action:none}aside{background:#fff0d8;padding:18px;border-radius:10px}h1{font-size:28px}</style><h1>MASt3R 动态过滤与稀疏建图验证</h1><aside>'+limits+'</aside><p>'+timing+'</p>'+table+(ROOT/'architecture.svg').read_text()+viewer+''.join(sections)+'</html>'
    body=body.replace('</style>','svg{max-width:100%;height:auto}</style>')
    (OUT/'report.html').write_text(body,encoding='utf-8')
    md='# 实际运行验证报告\n\n'+limits+'\n\n'+timing+'\n\n|配置|帧数|有效位姿率|ATE SE3 RMSE (m)|MASt3R (ms)|CNN (ms)|原生后端 (ms)|剔除候选点|局部 BA|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for m in metrics:md+=f'|{m["mode"]}|{m["frames"]}|{m["valid_fraction"]:.1%}|{m["ate_se3_rmse_m"]:.4f}|{m["mast3r_ms_mean"]:.1f}|{m["gate_ms_mean"]:.2f}|{m["backend_ms_mean"]:.1f}|{m["cnn_rejected_total"]}|{m["ba_calls"]}|\n'
    if len(metrics)==2:
        b,d=metrics
        conclusion='本次动态分支没有优于关闭过滤的对照，需要继续训练与标定，不能宣称精度提升。' if d['ate_se3_rmse_m']>=b['ate_se3_rmse_m'] or d['valid_fraction']<b['valid_fraction'] else '本次动态分支的轨迹指标有所改善，但单序列单次运行不能证明可泛化的提升。'
        md+='\n'+conclusion+'\n'
        md+=f'\n动态分支撤销已建地图点 {d["retired_map_points_total"]} 个。这是算法判定数量，不是人工验证的正确检测数。\n'
    md+='\n点云文件：`results/dynamic/dense_filtered_candidate.ply` 为候选融合点云，`dense_static_multiview.ply` 需要至少两个关键帧的体素支持，`sparse_map.ply` 为优化器地图点。两视角支持不等同于静态物体真值。轨迹只保存通过跟踪检查的帧，不把失败帧伪装成成功。\n'
    (ROOT/'验证报告.md').write_text(md,encoding='utf-8');print('REPORT_READY')
if __name__=='__main__':main()

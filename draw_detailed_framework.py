from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import html
ROOT=Path(__file__).resolve().parent/'figures';ROOT.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc'
class Figure:
 def __init__(self,w,h,title,subtitle):
  self.w=w;self.h=h;self.im=Image.new('RGB',(w,h),'#f4f7fc');self.d=ImageDraw.Draw(self.im);self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="#f4f7fc"/>'];self.text(45,22,title,38);self.text(45,80,subtitle,23,'#536783')
 def text(self,x,y,s,size=24,color='#18314d'):
  s=s.translate(str.maketrans({'ᵣ':'r','ₜ':'t','⑪':'(11)','⑫':'(12)','⑬':'(13)','⑭':'(14)','⑮':'(15)','⑯':'(16)'}))
  self.d.text((x,y),s,font=ImageFont.truetype(FONT,size),fill=color)
  self.svg.append(f'<text x="{x}" y="{y+size}" font-family="Microsoft YaHei,sans-serif" font-size="{size}" fill="{color}">{html.escape(s)}</text>')
 def box(self,x,y,w,h,title,lines,color='#e9f2ff'):
  self.d.rounded_rectangle((x,y,x+w,y+h),14,fill=color,outline='#6386ac',width=2);self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{color}" stroke="#6386ac" stroke-width="2"/>');self.text(x+16,y+14,title,27)
  for i,s in enumerate(lines):self.text(x+16,y+61+i*35,s,22)
 def arrow(self,pts,color='#486b92',width=4):
  self.d.line(pts,fill=color,width=width);x,y=pts[-1];a,b=pts[-2];dx=x-a;dy=y-b;l=max((dx*dx+dy*dy)**.5,1);ux,uy=dx/l,dy/l;p=[(x,y),(x-15*ux+7*uy,y-15*uy-7*ux),(x-15*ux-7*uy,y-15*uy+7*ux)];self.d.polygon(p,fill=color);self.svg.append('<polyline points="'+' '.join(f'{a},{b}' for a,b in pts)+f'" fill="none" stroke="{color}" stroke-width="{width}"/>');self.svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in p)+f'" fill="{color}"/>')
 def save(self,name):
  self.im.save(ROOT/(name+'.png'));(ROOT/(name+'.svg')).write_text(''.join(self.svg)+'</svg>',encoding='utf-8')

f=Figure(2700,1480,'MASt3R 动态过滤与稀疏 SLAM：已实现的详细框架','蓝色：学习前端及适配模块    绿色：复用 ORB-SLAM3 地图与优化    橙色：动态点处理    箭头表示数据依赖')
xs=[45,490,935,1380,1825,2270];bw=385
top=[('① 双帧输入与标定',['上一有效参考帧 Iᵣ','当前 RGB 图像 Iₜ + 内参 K','初始化等待两张不同图像','缩放 / 裁剪并同步调整 K']),('② MASt3R 稠密预测',['共享编码与双视图解码','点图 Pᵣ、Pₜ→ᵣ：H×W×3','描述子 Dᵣ、Dₜ：H×W×24','几何与描述子置信度']),('③ 候选与初步关联',['8×8 网格内按联合评分选点','纹理梯度 + 两类置信度','浮点 L2 匹配 / ratio 筛选','双向一致性 → 初步对应']),('④ 相机运动与尺度',['已有地图观测锚定深度尺度','初步 PnP-RANSAC','得到参考→当前变换 Tₜᵣ','只用于构建运动补偿线索']),('⑤ 有序点图特征融合',['RGB 3 + 归一化 XYZ 3','置信度 1 + 重投影残差 1','三维点一致性残差 1','对应支持度 1 → 共 10 通道']),('⑥ 轻量动态点 CNN',['1×1 通道投影','深度可分离卷积 ×3','1×1 概率头 + Sigmoid','输出 Pdyn：H×W×1'])]
for i,(t,l) in enumerate(top):f.box(xs[i],155,bw,245,t,l,'#fff0dc' if i==5 else '#e9f2ff')
for i in range(5):f.arrow([(xs[i]+bw,277),(xs[i+1],277)])
f.text(45,445,'最终定位链路：先过滤，再组织稀疏观测，最后优化位姿',28)
bottom=[('⑦ 风险过滤与检查',['Pdyn > 0.65：剔除候选点','置信度分位数 + 最低门槛','有效深度 / 有限值检查','无匹配支持区域按未知保留']),('⑧ 稀疏观测构建',['保留已跟踪点及地图 ID','新点按 3.5 cm 体素去重','像素位置 / 24 维描述子','世界三维坐标 / 关联 ID']),('⑨ 原生位姿估计',['已有 MapPoint 的 3D–2D 对应','地图点 PnP-RANSAC','Optimizer::PoseOptimization','检查内点数，确定有效位姿']),('⑩ 关键帧与稀疏地图',['Atlas / Map / KeyFrame','MapPoint 及观测关联','插入可信新点 / 更新共视关系','Optimizer 局部 BA']),('⑪ 稠密地图维护',['保存关键帧过滤后局部点云','用优化后关键帧位姿变换','4 cm 体素融合与颜色融合','另输出 ≥2 关键帧支持点']),('⑫ 输出与下一帧',['有效相机轨迹 Tcw','稀疏地图 / 稠密候选点云','多视角支持点云 / 日志','更新参考帧、点 ID 与地图坐标'])]
for i,(t,l) in enumerate(bottom):f.box(xs[i],510,bw,245,t,l,'#fff0dc' if i==0 else '#e7f5ec' if i>=2 else '#e9f2ff')
for i in range(5):f.arrow([(xs[i]+bw,632),(xs[i+1],632)])
f.arrow([(2462,400),(2462,425),(25,425),(25,632),(45,632)],'#c07a30')
f.box(45,845,800,220,'已建动态点的撤销支路',['已关联地图 ID + Pdyn > 0.85 + 充分匹配支持','重投影残差与三维不一致性同时提供证据','调用 MapPoint::SetBadFlag，撤销点与观测关联'],'#fff0dc')
f.arrow([(237,755),(237,845)],'#c07a30');f.arrow([(845,950),(1572,950),(1572,755)],'#c07a30')
f.box(935,845,800,220,'跟踪失败与反馈',['可用观测 / 内点不足：不插入地图、不保存为有效位姿','保留上一次有效参考帧，尝试后续更宽基线匹配','有效结果反馈到下一帧：参考图像、像素位置、地图 ID、3D、Tcw'],'#e9f2ff')
f.arrow([(2462,755),(2462,800),(1335,800),(1335,845)])
f.box(1825,845,830,220,'两条地图路径共享位姿优化结果',['定位：少量带描述子、观测关联的稀疏地图点','建图：关键帧的稠密过滤点，经优化位姿进行融合','多视角体素支持有助于减少瞬态点，但不等于静态真值'],'#e7f5ec')
f.text(45,1130,'坐标约定',29);f.text(45,1180,'MASt3R 第二张点图也位于参考相机坐标系：Xworld = inverse(Tcw,reference) × Xcurrent→reference',26)
f.text(45,1230,'初步 PnP 用于相机运动补偿；过滤后的地图点 PnP + 位姿优化用于最终定位，两者职责不同。',26)
f.box(45,1300,2610,135,'当前实现边界',['CNN 仅合成训练；单目 BA 仍可能尺度漂移；未适配 ORB 词袋回环、全局重定位、IMU 和完整历史稠密动态回溯。'],'#fff0dc');f.save('01_detailed_pipeline')

f=Figure(2700,720,'DynamicPointCNN：输入、卷积结构与过滤决策','在保留像素邻接关系的有序点图上执行二维卷积；参数总数 3,817；当前模型仅经过合成残差训练')
blocks=[('10 通道输入',['RGB：3；归一化 XYZ：3','几何置信度：1','重投影 / 三维残差：各 1','对应支持度：1']),('通道投影',['1×1 Conv','10 → 24','ReLU','保持 H×W 分辨率']),('可分离卷积块 1',['3×3 DW，dilation=1','24 → 24；ReLU','1×1 PW：24 → 32','ReLU']),('可分离卷积块 2',['3×3 DW，dilation=2','32 → 32；ReLU','1×1 PW：32 → 32','ReLU']),('可分离卷积块 3',['3×3 DW，dilation=4','32 → 32；ReLU','1×1 PW：32 → 24','ReLU']),('概率头与后处理',['1×1 Conv：24 → 1','Sigmoid → 动态风险概率','首帧 / 无支持区域：未知','候选过滤阈值：0.65'])]
for i,(t,l) in enumerate(blocks):f.box(xs[i],160,bw,250,t,l,'#fff0dc' if i==5 else '#e9f2ff')
for i in range(5):f.arrow([(xs[i]+bw,285),(xs[i+1],285)])
f.box(45,475,1260,180,'为什么需要相机运动补偿',['相机移动会使静态背景在图像中移动。用初步相机位姿解释刚体运动，','把未被该模型解释的残差及三维一致性线索交给 CNN，预测运动风险。'],'#e7f5ec')
f.box(1360,475,1295,180,'概率的使用范围',['当前候选点：高风险点不参与最终位姿估计和当前点云融合。','已建地图点：更高阈值并结合两类几何证据，才撤销地图观测关联。'],'#fff0dc');f.save('02_cnn_detail')

f=Figure(2700,930,'原始 ORB-SLAM3 与新框架：替换、复用及未适配部分','上：原始视觉 SLAM 主链路概览    下：当前工程已实现的学习前端与原生优化后端')
old=[('图像输入',['单目 / 双目 / RGB-D','可选 IMU 输入']),('ORB 特征提取',['图像金字塔 / FAST 角点','方向与二进制描述子']),('特征匹配与跟踪',['Hamming 匹配','运动模型 / 局部地图跟踪']),('关键帧与局部建图',['三角化或深度生成地图点','共视关系 / 地图点维护']),('几何优化',['位姿优化','局部 BA / 图优化']),('回环与输出',['ORB 词袋回环 / 重定位','轨迹与稀疏地图'])]
new=[('RGB 双帧输入',['相机内参 K','等待双帧完成初始化']),('MASt3R 学习前端',['稠密三维点图','浮点描述子与置信度']),('卷积动态点过滤',['相机运动补偿线索','10 通道 CNN 风险预测']),('稀疏化与关联',['网格选点 + 体素去重','L2 匹配 + 持久地图 ID']),('ORB-SLAM3 后端',['保留 Atlas / Map / KeyFrame','MapPoint / 位姿优化 / 局部 BA']),('稠密与稀疏输出',['优化位姿驱动稠密融合','轨迹 / 稀疏点 / 稠密点云'])]
for row,items in enumerate((old,new)):
 y=160+row*340
 for i,(t,l) in enumerate(items):f.box(xs[i],y,bw,170,t,l,'#e7f5ec' if row and i==4 else '#e9f2ff' if row else '#edf0f4')
 for i in range(5):f.arrow([(xs[i]+bw,y+85),(xs[i+1],y+85)])
f.text(490,385,'替换：ORB 检测 / 二进制匹配 / 基于 ORB 的观测生成 → 稠密几何引导的动态过滤与稀疏观测',27,'#a56125')
f.box(45,745,2610,135,'工程范围说明',['新增 MASt3RSystem 适配前端直接驱动原生地图与优化；原版 Tracking / LocalMapping / LoopClosing 线程未启动。词袋回环、IMU 等尚未适配。'],'#fff0dc');f.save('03_original_vs_new')
print(ROOT)

# AWS TextVQA 全验证集工具包

适用于已经在 AWS 上跑通 pilot10 的项目：~/11785-visionzip。
使用已有 .venv、activate-project.sh、visionzip_eval.py、data/textvqa/source.json。
不需要重新安装模型；本工具不会自动停止 EC2，不会创建新实例。
本地已检查语法和预测完整性验证逻辑，尚未在你的远程 GPU 上执行端到端验证。

## 上传（在 Mac 新开的本地终端中执行）

先下载 aws-textvqa-full.zip 到 Mac 的 Downloads，再执行：

```bash
scp -i ~/Downloads/11785-visionzip.pem \
  ~/Downloads/aws-textvqa-full.zip \
  ubuntu@52.14.34.48:~/11785-visionzip/
```

如实例公网地址变化，使用 AWS 页面当前地址。不要把这个 scp 命令粘贴到远程 Ubuntu 终端。

## 解压和启动（在已连接的 AWS 远程终端执行）

```bash
source ~/11785-visionzip/activate-project.sh
python -m zipfile -e aws-textvqa-full.zip .
ls aws-textvqa-full
nohup bash aws-textvqa-full/run_full.sh > logs/full-textvqa.log 2>&1 &
echo $!
tail -f logs/full-textvqa.log
```

Ctrl+C 只退出 tail，不停止后台作业。flock 会拒绝同时启动第二份 full 作业。
不要 Stop EC2，否则后台作业也会停止。关闭 SSH 后，nohup 作业可继续运行。

## 作业内容

1. 使用 pilot 保存的数据集版本，流式准备全部 5000 问答、3166 图片。
   保留同一图片的多个问题；不再按图片去重问题。
2. 顺序运行原始 LLaVA 与 VisionZip-64；两组参数相同，避免同时占用显存。
3. 分别检查 5000 条预测与问题 ID、prompt 对齐；空预测保留并按原评分器处理。
4. 调用 LLaVA 的 TextVQA 评分入口。
5. 导出逐问题分数、按图像配对的 bootstrap 95% 差值区间（2000 次，seed 785）、示例案例和部分案例图片。

完整覆盖不等于已经对齐原论文评测：本作业沿用项目 supplied-OCR prompt，未核对 LLaVA eval.zip 的原始 prompt/OCR 预处理。
正式报告写明该区别。图像保存为 RGB PNG，问题和 OCR 内部空白折叠，以确保评分入口正确解析三行提示。
计时日志用于估算作业进度，不是受控 TTFT/延迟实验。
bootstrap 区间描述该评测分布上的抽样不确定性，不覆盖硬件/实现等系统误差。

## 输出

运行目录路径保存在 logs/latest-full-run.txt。
每次作业使用新目录 outputs/textvqa-full-时间-PID，已有预测不会被覆盖。
该版本不自动续跑失败的推理；若中断，不要直接重启整个流程，先检查最后日志。

```bash
cat logs/latest-full-run.txt
tail -n 40 logs/full-textvqa.log
```

看到 ALL DONE 及对应目录内 COMPLETE 文件，表示两个配置和分析都完成。
主要文件：
- summary.json: 两组精度、差值、置信区间。
- baseline-score.log / visionzip64-score.log: 原评分器输出。
- baseline.jsonl / visionzip64.jsonl: 全部模型预测。
- per_question.csv: 逐问题软分数、预测和参考答案。
- example_cases.json / case_images: 挑选的改善、退化、共同错误案例。
- requirements.txt、gpu-info.txt、run-config.json、代码 commit、数据版本等复现记录。

## 备份与停止

在 AWS 远程终端：

```bash
cd ~/11785-visionzip
tar -czf textvqa-full-results.tar.gz outputs/textvqa-full-* logs
```

在 Mac 本地终端：

```bash
scp -i ~/Downloads/11785-visionzip.pem \
  ubuntu@52.14.34.48:~/11785-visionzip/textvqa-full-results.tar.gz \
  ~/Downloads/
```

确认备份后，如果不继续做 GPU 实验，到 AWS 控制台 Stop instance；EBS 仍计费。
保存旧的 pilot10 结果，不将它当作独立于 full validation 的重复实验。

## Midreport 要点

完整实验是 LLaVA 与 VisionZip baseline 对照；跨层扩展仍应标为下一阶段工作。
核心材料：4–5 页指定模板，问题/目标、相关工作、方法、数据与参数、自己运行的结果、挑战和2–3项下一步。
不要把 token 数减少88.9%写成运行时间或总显存减少88.9%。
没有训练，因此说明所有参数冻结、无新增 loss/optimizer；无需伪造训练曲线。
若要写加速结论，另做相同输入和输出控制、warm-up、同步、重复测量的效率实验。

参考：
- https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
- https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/eval_textvqa.py
- https://textvqa.org/challenge/

# 在已有 AWS 项目补跑 GQA

目标：冻结 LLaVA-1.5-7B，比较原模型（576 视觉 tokens）与免训练 VisionZip-64（54 dominant + 10 contextual）。本工具不实现跨层扩展，也不微调 projector。

本工具适配你已经跑通 TextVQA 的 Ubuntu / Python 3.10 `.venv` / L4 环境。无需重装依赖。使用已有 PyTorch、Transformers、huggingface_hub、pyarrow、Pillow、numpy、tqdm。不要在 Mac 的 conda base 中安装这些包。

## 数据和协议

- 使用 LLaVA 官方 GQA 脚本采用的 `testdev_balanced`，共 12,578 问题、398 图片。
- 数据来自 `lmms-lab-encoder/GQA` 的固定 revision `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8`。
- 仅下载 `testdev_balanced_instructions` 和 `testdev_balanced_images` 两个 parquet 文件，合计 67,738,080 bytes，约 68 MB。不会下载全套训练集。
- 保留镜像中的原始图片编码字节；问题末尾加 `Answer the question using a single word or phrase.`，不添加 OCR。
- 两组同为 FP16、batch 1、greedy、vicuna_v1、max_new_tokens 128，独立进程顺序运行。
- GQA 采用单参考答案的二值准确率，不是 TextVQA 的十参考答案软评分。
- 转换预测使用 LLaVA 的 `text.rstrip('.').lower()`；再用其文档链接的 GQA 评分器交叉核对 Accuracy。
- 该评分器关闭了部分需要额外资源的指标，其日志中的 Validity/Plausibility 零值不是测量结果。报告只使用 Accuracy 和实际计算的分类准确率。
- 使用镜像重建输入，没有逐字节对齐原论文 eval.zip 的问题文件，因此还不能宣称完整复现所有原论文协议细节。

## 1. 把工具包传到服务器（在 Mac 本地终端）

启动你原来的 EC2 实例，等状态检查通过。停止后再启动，公网 IP 可能已改变，请从 EC2 控制台复制当前 Public IPv4 address。

下面把 `当前公网IP` 换成真实地址；不要保留中文占位符。

```bash
scp -i ~/Downloads/11785-visionzip.pem \
  /Users/torry/Documents/ChatGPT/11785/aws-gqa.zip \
  ubuntu@当前公网IP:~/11785-visionzip/
ssh -i ~/Downloads/11785-visionzip.pem ubuntu@当前公网IP
```

若 SSH 超时，检查实例状态和安全组 SSH 来源是否仍为你当前 My IP。

## 2. 解压、激活、检查（下面均在服务器终端）

```bash
source ~/11785-visionzip/activate-project.sh
python -m zipfile -e aws-gqa.zip .
pwd
which python
nvidia-smi
df -h /
```

工作目录应为 `/home/ubuntu/11785-visionzip`，Python 路径应含 `.venv/bin/python`。确保至少 2 GiB 可用空间。

## 3. 先做 20 题检查

```bash
set -o pipefail
bash aws-gqa/run.sh smoke 2>&1 | tee logs/gqa-smoke.log
```

看到 `GQA DATA CHECK PASSED`、两组 `official evaluator agrees`、最后 `ALL DONE (smoke)`，说明整条管线通过。两组投影 token 数会被检查为 576 和 64。20 题分数仅用于检查，不作为正式实验结论。

第一次需要下载数据，准备时间取决于网络。短跑还需要分别加载两次模型，一般是分钟量级；实际以日志为准。失败时不要执行全量，先查看错误，不要重装整个环境。

## 4. 运行完整两组

```bash
nohup bash aws-gqa/run.sh full > logs/full-gqa.log 2>&1 &
echo $!
tail -f logs/full-gqa.log
```

`Ctrl+C` 只结束 `tail -f` 的查看，不会停止后台 `nohup` 任务。成功启动后可断开 SSH；EC2 必须保持 Running。关闭电脑不会停云端任务，也不会停止计费。

按你已完成的 TextVQA 两组循环耗时（共约72分钟/5000题）仅做比例估计，GQA 两组 12578 题约需3小时。GQA提示更短、答案长度和机器负载可能不同，建议预留2–4小时，不能保证75分钟。先看 GQA 的实际进度估计。

这个脚本不会自动停止 EC2。不要重复启动同一任务；锁会阻止本工具和旧 TextVQA 全量工具同时占用 GPU。

## 5. 查看完成状态和分数

重新连接服务器后：

```bash
source ~/11785-visionzip/activate-project.sh
tail -n 50 logs/full-gqa.log
RUN_DIR=$(cat logs/latest-gqa-full-run.txt)
cat "$RUN_DIR/summary.json"
```

以 `ALL DONE (full)` 和 `COMPLETE` 标记为完成；不要仅凭看到某一组 100% 就停止实例。

输出在新的 `outputs/gqa-full-时间-PID/` 中，包含两组原始预测、官方格式预测与评分日志、逐题CSV、汇总JSON、按图像配对bootstrap区间、配置、代码版本和依赖记录。既有 TextVQA 结果不会被覆盖。

单次 inference-process 耗时包含模型加载；不是受控 TTFT 或速度基准。GPU信息文件是推理前快照，不能作为峰值显存结果。

## 6. 打包、下载、Stop

服务器终端：

```bash
python aws-gqa/pack_results.py
```

另开 Mac 本地终端，使用实例当前公网IP：

```bash
scp -i ~/Downloads/11785-visionzip.pem \
  ubuntu@当前公网IP:~/11785-visionzip/gqa-full-results.tar.gz \
  ~/Downloads/
```

确认本地文件下载成功后，可发给我检查，并在 AWS 中 Stop 实例、确认状态变为 Stopped。EBS 磁盘保留期间仍计费。

## 来源

- LLaVA Evaluation: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#gqa
- LLaVA GQA runner: https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/gqa.sh
- LLaVA prediction converter: https://github.com/haotian-liu/LLaVA/blob/main/scripts/convert_gqa_for_eval.py
- LLaVA 文档链接的 GQA evaluator: https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187
- 固定镜像数据: https://huggingface.co/datasets/lmms-lab-encoder/GQA/tree/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8
- VisionZip 官方接入方式: https://github.com/dvlab-research/VisionZip#evaluation

本地验证覆盖真实12,578题/398图的数据准备、图片字节与哈希、模拟预测的官方评分交叉核对、缺失ID/错误提示拒绝和配对分析；未在本地Mac上运行CUDA推理。真实GPU推理由服务器20题检查验证。

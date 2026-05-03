# Simple Captioner

This is a fork of [Olli S.](https://github.com/o-l-l-i) [simple-captioner](https://github.com/o-l-l-i/simple-captioner); check the original repository for the README, and thanks the original author for the hard work.

This fork add:
* updated dependencies (with side effects, see the [known bugs and issues](BUGS.md))
* some optimizations (credits to Clau.de):
  * can batch images; this helped me to improve the throughput by one order of magnitude on my hardware

---

## Usage Notes

```sh
# useful environment variables:
export HF_HUB_OFFLINE=1 GRADIO_SERVER_NAME="0.0.0.0"

uv run python app.py

# try another Python version with an optional dependency
uv run --extra flash4 --with scalene --python 3.10 python app.py
```

## This works for me™

* GeForce RTX 3060 12 GiB VRAM
* Debian GNU/Linux 13 (trixie)
* 3 vCPU, 19 GiB RAM
* Python 3.14

Aggressive parameters:
* Qwen/Qwen3-VL-4B-Instruct, 4-bit, eager
* 🧾 Max Tokens: 4096
* 🧵 Prefetch Workers: 2 (of 1 advised)
* 📦 Batch Size: 10 (of 4 advised)

#!/usr/bin/env python3
"""
sim_main_manager.py

父进程管理器 - 负责启动和重启 Isaac Sim 子进程

架构：
- 父进程：监控 Redis reset 命令，管理子进程生命周期
- 子进程：运行实际的 Isaac Sim 仿真 (sim_main_recreate.py)
- Reset 流程：父进程 kill 子进程 → 等待清理完成 → 重新启动子进程
"""

import sys
import os
import subprocess
import time
import signal
import redis
import json

def main():
    # 获取命令行参数（传递给子进程）
    child_args = sys.argv[1:]

    print("="*80)
    print("🎮 Isaac Sim Manager - Parent Process")
    print("="*80)
    print(f"PID: {os.getpid()}")
    print(f"Child command: python sim_main_recreate.py {' '.join(child_args)}")
    print("="*80 + "\n")

    # 连接 Redis
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        return 1

    child_process = None
    restart_count = 0

    def cleanup_handler(sig, frame):
        """处理 Ctrl+C"""
        print("\n\n🛑 Received interrupt signal, cleaning up...")
        if child_process and child_process.poll() is None:
            print(f"Terminating child process (PID: {child_process.pid})...")
            child_process.terminate()
            try:
                child_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing child process...")
                child_process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    while True:
        # 启动子进程
        restart_count += 1
        print(f"\n{'='*80}")
        print(f"🚀 Starting Isaac Sim child process (restart #{restart_count})")
        print(f"{'='*80}\n")

        # 添加环境变量标记，告诉子进程它是被管理的
        env = os.environ.copy()
        env['ISAAC_SIM_MANAGED'] = '1'

        child_process = subprocess.Popen(
            [sys.executable, 'sim_main_recreate.py'] + child_args,
            env=env,
            stdout=None,  # 继承父进程的 stdout
            stderr=None,  # 继承父进程的 stderr
        )

        print(f"✅ Child process started (PID: {child_process.pid})")

        # 监控子进程和 Redis reset 命令
        last_check_time = time.time()
        check_interval = 0.5  # 每 0.5 秒检查一次

        while True:
            # 检查子进程是否还在运行
            if child_process.poll() is not None:
                exit_code = child_process.returncode
                print(f"\n⚠️ Child process exited with code {exit_code}")

                if exit_code != 0:
                    print(f"❌ Child process crashed! Waiting 3 seconds before restart...")
                    time.sleep(3)

                break  # 退出内层循环，重新启动子进程

            # 定期检查 Redis reset 命令
            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                last_check_time = current_time

                try:
                    reset_trigger_raw = redis_client.get("isaac_reset_trigger_manager")

                    if reset_trigger_raw:
                        reset_trigger = json.loads(reset_trigger_raw)
                        reset_category = reset_trigger.get("reset_category", "0")

                        if reset_category == "3":
                            print(f"\n{'='*80}")
                            print("🔄 RESET COMMAND RECEIVED (category 3)")
                            print(f"{'='*80}")

                            # 清除 trigger
                            redis_client.delete("isaac_reset_trigger_manager")

                            # Kill 子进程
                            print(f"🛑 Terminating child process (PID: {child_process.pid})...")
                            child_process.terminate()

                            # 等待子进程退出（最多 10 秒）
                            try:
                                child_process.wait(timeout=10)
                                print("✅ Child process terminated gracefully")
                            except subprocess.TimeoutExpired:
                                print("⚠️ Child process did not exit, force killing...")
                                child_process.kill()
                                child_process.wait()
                                print("✅ Child process killed")

                            # 等待 GPU 资源释放
                            print("⏳ Waiting for GPU resources to be released...")
                            time.sleep(2)

                            # 发送重置完成信号
                            try:
                                reset_complete_signal = {
                                    "status": "complete",
                                    "timestamp": int(time.time() * 1000)
                                }
                                redis_client.set("isaac_reset_complete_unitree_g1_with_hands",
                                               json.dumps(reset_complete_signal))
                                redis_client.expire("isaac_reset_complete_unitree_g1_with_hands", 5)
                                print("✅ Reset complete signal sent via Redis")
                            except Exception as e:
                                print(f"⚠️ Failed to send reset complete signal: {e}")

                            break  # 退出内层循环，重新启动子进程

                except redis.ConnectionError:
                    pass  # Redis 暂时不可用，忽略
                except Exception as e:
                    print(f"⚠️ Error checking reset trigger: {e}")

            # 短暂休眠，避免 CPU 占用过高
            time.sleep(0.1)

if __name__ == "__main__":
    sys.exit(main())

import subprocess
import sys
import os
import re
import datetime
from collections import Counter

# 配置区域
# 定义需要高亮关注的关键词（不区分大小写）
ERROR_KEYWORDS = [
    "failed", "error", "dependency", "timed out", "panic",
    "segfault", "corruption", "unreachable", "critical"
]

# 定义一些通常可以忽略的噪音关键词（根据实际情况调整）
IGNORE_KEYWORDS = [
    "predictable naming", # 网卡命名相关
    "error changing net interface name"
]

class BootAnalyzer:
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"./boot_logs_{self.timestamp}"

    def check_root(self):
        """检查是否以 root/sudo 权限运行"""
        if os.geteuid() != 0:
            print("❌ 错误: 此脚本需要 root 权限才能查看完整日志。")
            print("请使用: sudo python3 boot_analyzer.py")
            sys.exit(1)

    def run_command(self, command):
        """运行 shell 命令并返回输出"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors='replace' # 防止编码错误导致脚本崩溃
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error running command: {e}"

    def get_boot_list(self):
        """获取系统启动列表"""
        print("🔍 正在读取启动历史...")
        return self.run_command("journalctl --list-boots --no-pager")

    def analyze_log_content(self, log_content, source_name):
        """分析日志内容，提取错误信息"""
        issues = []
        lines = log_content.splitlines()

        for line in lines:
            line_lower = line.lower()
            # 检查是否有错误关键词
            if any(keyword in line_lower for keyword in ERROR_KEYWORDS):
                # 检查是否在忽略列表中
                if not any(ignore in line_lower for ignore in IGNORE_KEYWORDS):
                    issues.append(line.strip())

        return issues

    def save_to_file(self, filename, content):
        """保存内容到文件"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")

    def run(self):
        self.check_root()
        self.print_header("Ubuntu 启动故障诊断工具 (v1.0)")

        # 1. 获取启动列表
        boot_list = self.get_boot_list()
        print(boot_list)

        # 2. 询问用户要分析哪次启动
        print("\n👉 提示: '-0' 是当前启动，'-1' 是上一次启动，以此类推。")
        boot_id = input("请输入要分析的启动 ID (默认为 -1, 即上一次): ").strip()
        if not boot_id:
            boot_id = "-1"

        print(f"\n⏳ 正在提取 Boot {boot_id} 的日志，请稍候...")

        # 3. 获取各类日志
        # Systemd Journal (所有日志)
        journal_log = self.run_command(f"journalctl -b {boot_id} --no-pager")

        # Kernel Ring Buffer (如果分析的是当前启动，用dmesg；如果是历史启动，用 journalctl -k)
        if boot_id == "0":
            dmesg_log = self.run_command("dmesg -T")
        else:
            dmesg_log = self.run_command(f"journalctl -k -b {boot_id} --no-pager")

        # Failed Services (systemctl list-units --failed 只能看当前的，历史的需从日志分析)

        # 4. 执行分析
        journal_issues = self.analyze_log_content(journal_log, "Systemd Journal")
        dmesg_issues = self.analyze_log_content(dmesg_log, "Kernel/Dmesg")

        # 5. 统计高频错误
        issue_counter = Counter(journal_issues)
        top_issues = issue_counter.most_common(5)

        # 6. 输出报告
        self.print_header("📊 分析报告摘要")
        print(f"分析对象: Boot {boot_id}")
        print(f"日志保存目录: {self.log_dir}")
        print(f"发现潜在问题总数: {len(journal_issues)}")
        print(f"其中内核级问题数: {len(dmesg_issues)}")

        if top_issues:
            print("\n🔥 出现频率最高的 Top 5 错误:")
            for issue, count in top_issues:
                # 截断过长的行以便显示
                display_issue = (issue[:80] + '...') if len(issue) > 80 else issue
                print(f" [{count}次] {display_issue}")
        else:
            print("\n✅ 未检测到明显的重复错误模式。")

        print("\n💡 关键错误样本 (最近 5 条):")
        for issue in journal_issues[-5:]:
            print(f" -> {issue}")

        # 7. 保存原始数据
        self.save_to_file("full_journal.log", journal_log)
        self.save_to_file("dmesg_kernel.log", dmesg_log)
        self.save_to_file("analysis_report.txt", "\n".join(journal_issues))

        print(f"\n✅ 完成！原始日志和详细错误列表已保存至文件夹: {self.log_dir}")
        print("请查看 'analysis_report.txt' 获取完整错误清单。")

if __name__ == "__main__":
    analyzer = BootAnalyzer()
    analyzer.run()

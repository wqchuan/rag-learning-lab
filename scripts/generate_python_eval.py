"""Generate the deterministic Python 3.14.7 learning corpus and schema-v2 review set."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpora/python-3.14.7"
EVAL = ROOT / "data/eval/python_docs_v2.json"
REVIEW = ROOT / "data/eval/python_docs_v2_review.csv"
VERSION = "Python 3.14.7"
LICENSE = "PSF License Version 2"
RETRIEVED_AT = "2026-09-19"

TOPICS = [
    ("tutorial-introduction", "Python 教程概览", "https://docs.python.org/zh-cn/3.14/tutorial/introduction.html", [
        ("除法运算", "运算符 / 总是返回浮点数，// 执行向下取整除法。", ["浮点数", "向下取整"]),
        ("幂运算", "Python 使用 ** 运算符计算乘方。", ["**", "乘方"]),
        ("字符串索引", "字符串支持从零开始的索引，负索引从末尾计数。", ["从零", "负索引"]),
        ("字符串不可变性", "Python 字符串不可变，不能直接修改某个索引位置。", ["不可变"]),
    ]),
    ("tutorial-controlflow", "流程控制", "https://docs.python.org/zh-cn/3.14/tutorial/controlflow.html", [
        ("range 终点", "range 的终点不包含在生成的序列中。", ["不包含"]),
        ("for 迭代", "for 语句按顺序迭代序列中的项目。", ["按顺序", "项目"]),
        ("match 用途", "match 语句把一个值与若干模式依次比较。", ["模式", "比较"]),
        ("函数默认值", "函数默认参数值只在定义时求值一次。", ["定义时", "一次"]),
    ]),
    ("tutorial-datastructures", "数据结构", "https://docs.python.org/zh-cn/3.14/tutorial/datastructures.html", [
        ("列表追加", "list.append(x) 把一个项目添加到列表末尾。", ["append", "末尾"]),
        ("列表推导式", "列表推导式用简洁方式从可迭代对象创建列表。", ["可迭代对象", "列表"]),
        ("元组", "元组是不可变序列，通常用圆括号书写。", ["不可变", "圆括号"]),
        ("集合", "集合是无序且不含重复元素的容器。", ["无序", "不含重复"]),
    ]),
    ("tutorial-modules", "模块", "https://docs.python.org/zh-cn/3.14/tutorial/modules.html", [
        ("模块定义", "模块是包含 Python 定义和语句的文件。", ["定义", "语句", "文件"]),
        ("模块名称", "模块内部的 __name__ 变量保存模块名称。", ["__name__", "模块名称"]),
        ("导入缓存", "同一解释器会话中模块通常只导入一次。", ["一次", "解释器会话"]),
        ("包作用", "包通过带点号的模块名组织 Python 模块。", ["带点号", "组织"]),
    ]),
    ("tutorial-inputoutput", "输入与输出", "https://docs.python.org/zh-cn/3.14/tutorial/inputoutput.html", [
        ("格式化字符串", "f 字符串在字符串前加 f，并在花括号中写表达式。", ["f", "花括号", "表达式"]),
        ("repr", "repr() 生成供解释器读取的对象表示。", ["解释器", "对象表示"]),
        ("文件打开", "open() 返回文件对象，常用参数包括文件名和打开模式。", ["文件对象", "打开模式"]),
        ("with 文件", "with 语句会在代码块结束后正确关闭文件。", ["关闭文件", "代码块"]),
    ]),
    ("tutorial-errors", "错误与异常", "https://docs.python.org/zh-cn/3.14/tutorial/errors.html", [
        ("语法错误", "语法错误会由解析器指出出错行和位置。", ["解析器", "位置"]),
        ("异常处理", "try 语句的 except 子句用于处理匹配的异常。", ["try", "except"]),
        ("finally", "finally 子句无论是否发生异常通常都会执行。", ["finally", "都会执行"]),
        ("raise", "raise 语句允许程序员显式触发指定异常。", ["raise", "触发"]),
    ]),
    ("tutorial-classes", "类", "https://docs.python.org/zh-cn/3.14/tutorial/classes.html", [
        ("类定义", "class 关键字用于创建新的类定义。", ["class", "类定义"]),
        ("实例属性", "实例属性通常通过 self.name 的形式赋值。", ["self", "赋值"]),
        ("继承", "派生类定义可在类名后的括号中写入基类。", ["派生类", "基类"]),
        ("迭代器", "实现 __iter__ 和 __next__ 可定义迭代器行为。", ["__iter__", "__next__"]),
    ]),
    ("tutorial-stdlib", "标准库概览", "https://docs.python.org/zh-cn/3.14/tutorial/stdlib.html", [
        ("os 模块", "os 模块提供与操作系统交互的函数。", ["os", "操作系统"]),
        ("shutil 模块", "shutil 模块提供更易用的文件和目录管理操作。", ["shutil", "目录管理"]),
        ("glob 模块", "glob 模块使用通配符搜索文件列表。", ["glob", "通配符"]),
        ("sys argv", "sys.argv 以列表形式保存命令行参数。", ["sys.argv", "命令行参数"]),
    ]),
    ("library-pathlib", "pathlib", "https://docs.python.org/zh-cn/3.14/library/pathlib.html", [
        ("Path 对象", "pathlib.Path 用面向对象方式表示文件系统路径。", ["Path", "文件系统路径"]),
        ("路径拼接", "Path 对象可使用 / 运算符拼接路径组件。", ["/", "路径组件"]),
        ("读取文本", "Path.read_text() 打开文件、读取文本并关闭文件。", ["read_text", "关闭文件"]),
        ("路径存在", "Path.exists() 判断路径是否指向现有文件或目录。", ["exists", "文件或目录"]),
    ]),
    ("library-json", "json", "https://docs.python.org/zh-cn/3.14/library/json.html", [
        ("序列化", "json.dumps() 把 Python 对象序列化为 JSON 字符串。", ["dumps", "JSON 字符串"]),
        ("反序列化", "json.loads() 从 JSON 文档反序列化 Python 对象。", ["loads", "Python 对象"]),
        ("文件写入", "json.dump() 把对象以 JSON 格式写入文件对象。", ["dump", "文件对象"]),
        ("中文保留", "ensure_ascii=False 可让非 ASCII 字符原样输出。", ["ensure_ascii", "False"]),
    ]),
    ("library-sqlite3", "sqlite3", "https://docs.python.org/zh-cn/3.14/library/sqlite3.html", [
        ("连接数据库", "sqlite3.connect() 打开到 SQLite 数据库的连接。", ["connect", "连接"]),
        ("参数绑定", "SQL 查询应使用占位符绑定值，以避免字符串拼接。", ["占位符", "绑定"]),
        ("提交事务", "Connection.commit() 提交当前事务。", ["commit", "事务"]),
        ("行访问", "sqlite3.Row 可让查询结果按列名访问。", ["Row", "列名"]),
    ]),
    ("library-argparse", "argparse", "https://docs.python.org/zh-cn/3.14/library/argparse.html", [
        ("解析器", "ArgumentParser 对象保存命令行参数的定义。", ["ArgumentParser", "命令行参数"]),
        ("添加参数", "add_argument() 告诉解析器程序接受哪些参数。", ["add_argument", "接受"]),
        ("解析参数", "parse_args() 解析参数并返回命名空间。", ["parse_args", "命名空间"]),
        ("帮助文本", "argparse 会根据参数定义自动生成帮助和用法信息。", ["自动生成", "帮助"]),
    ]),
    ("library-logging", "logging", "https://docs.python.org/zh-cn/3.14/library/logging.html", [
        ("日志级别", "标准日志级别包括 DEBUG、INFO、WARNING、ERROR 和 CRITICAL。", ["DEBUG", "CRITICAL"]),
        ("默认级别", "basicConfig() 未指定 level 时默认阈值为 WARNING。", ["WARNING", "阈值"]),
        ("记录器", "logging.getLogger(__name__) 可创建以模块名命名的记录器。", ["getLogger", "__name__"]),
        ("延迟格式化", "日志调用可把格式字符串和参数分开传入。", ["格式字符串", "参数"]),
    ]),
    ("library-unittest", "unittest", "https://docs.python.org/zh-cn/3.14/library/unittest.html", [
        ("测试用例", "测试类通常继承 unittest.TestCase。", ["TestCase", "继承"]),
        ("断言", "TestCase 提供 assertEqual 等断言方法。", ["assertEqual", "断言"]),
        ("测试发现", "unittest 支持按文件名模式自动发现测试模块。", ["自动发现", "文件名模式"]),
        ("准备方法", "setUp() 在每个测试方法运行前执行。", ["setUp", "运行前"]),
    ]),
    ("library-venv", "venv", "https://docs.python.org/zh-cn/3.14/library/venv.html", [
        ("创建环境", "python -m venv ENV_DIR 可创建虚拟环境。", ["python -m venv", "虚拟环境"]),
        ("环境隔离", "虚拟环境拥有独立于基础环境的已安装包集合。", ["独立", "包"]),
        ("激活作用", "激活脚本会把虚拟环境的可执行目录放到 PATH 前部。", ["PATH", "前部"]),
        ("可移植性", "虚拟环境通常不可移植，应在目标位置重新创建。", ["不可移植", "重新创建"]),
    ]),
    ("library-dataclasses", "dataclasses", "https://docs.python.org/zh-cn/3.14/library/dataclasses.html", [
        ("装饰器", "@dataclass 根据带类型注解的字段生成特殊方法。", ["@dataclass", "类型注解"]),
        ("默认工厂", "field(default_factory=...) 为可变默认值创建新对象。", ["default_factory", "新对象"]),
        ("冻结实例", "frozen=True 模拟只读的冻结实例。", ["frozen", "只读"]),
        ("转换字典", "dataclasses.asdict() 递归地把数据类实例转换为字典。", ["asdict", "字典"]),
    ]),
    ("library-typing", "typing", "https://docs.python.org/zh-cn/3.14/library/typing.html", [
        ("类型提示", "类型提示主要供类型检查器、IDE 和其他工具使用。", ["类型检查器", "IDE"]),
        ("运行时", "Python 运行时不强制执行函数和变量的类型注解。", ["不强制", "运行时"]),
        ("Protocol", "Protocol 可定义结构化子类型所需的方法和属性。", ["Protocol", "结构化子类型"]),
        ("TypedDict", "TypedDict 描述具有特定字符串键的字典类型。", ["TypedDict", "字符串键"]),
    ]),
    ("library-asyncio", "asyncio", "https://docs.python.org/zh-cn/3.14/library/asyncio.html", [
        ("用途", "asyncio 用 async/await 语法编写并发代码。", ["async", "await", "并发"]),
        ("运行入口", "asyncio.run() 运行协程并管理事件循环。", ["asyncio.run", "事件循环"]),
        ("任务", "asyncio.create_task() 把协程包装为任务并调度执行。", ["create_task", "任务"]),
        ("等待多个", "asyncio.gather() 并发运行多个可等待对象并汇总结果。", ["gather", "汇总"]),
    ]),
    ("library-concurrent-futures", "concurrent.futures", "https://docs.python.org/zh-cn/3.14/library/concurrent.futures.html", [
        ("高层接口", "concurrent.futures 提供异步执行可调用对象的高层接口。", ["异步执行", "高层接口"]),
        ("线程池", "ThreadPoolExecutor 使用线程池执行调用。", ["ThreadPoolExecutor", "线程池"]),
        ("进程池", "ProcessPoolExecutor 使用进程池并受多进程限制。", ["ProcessPoolExecutor", "进程池"]),
        ("Future", "Future 对象表示异步计算最终得到的结果。", ["Future", "结果"]),
    ]),
    ("library-hashlib", "hashlib", "https://docs.python.org/zh-cn/3.14/library/hashlib.html", [
        ("哈希算法", "hashlib 提供 SHA-2、SHA-3 等安全哈希算法。", ["SHA-2", "SHA-3"]),
        ("输入类型", "哈希对象的 update() 接受字节类对象。", ["update", "字节"]),
        ("十六进制摘要", "hexdigest() 以十六进制字符串返回摘要。", ["hexdigest", "十六进制"]),
        ("文件摘要", "hashlib.file_digest() 可计算文件对象内容的摘要。", ["file_digest", "文件对象"]),
    ]),
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    EVAL.parent.mkdir(parents=True, exist_ok=True)
    documents, cases, review_rows = [], [], []
    for topic_index, (slug, title, url, facts) in enumerate(TOPICS):
        path = f"{slug}.md"
        text = f"# {title}\n\n" + "\n".join(sentence for _, sentence, _ in facts) + "\n"
        (CORPUS / path).write_text(text, encoding="utf-8", newline="\n")
        doc_id = digest(path)[:16]
        documents.append({
            "doc_id": doc_id, "path": path, "title": title, "source_url": url,
            "source_version": VERSION, "license": LICENSE,
            "retrieved_at": RETRIEVED_AT, "content_hash": digest(text),
            "privacy": "public", "curation": "concise original summary linked to official documentation",
        })
        split = "dev" if topic_index < 12 else "test"
        answer_count = 3 if topic_index < 16 else 4
        for fact_index, (label, sentence, keywords) in enumerate(facts[:answer_count]):
            start = text.index(sentence)
            case_id = f"py314-{topic_index + 1:02d}-{fact_index + 1}"
            question = f"在 {title} 中，{label}是怎样规定或使用的？"
            evidence = [{
                "doc_id": doc_id, "path": path, "source_version": VERSION,
                "start_char": start, "end_char": start + len(sentence),
                "evidence_hash": digest(sentence), "required": True,
            }]
            cases.append({
                "id": case_id, "question": question, "family_id": slug,
                "type": "exact_term" if fact_index == 1 else "direct_fact",
                "split": split, "answerable": True,
                "expected_keywords": keywords, "evidence": evidence,
                "review_status": "pending_human",
            })
            review_rows.append([case_id, split, slug, question, "yes", sentence, "", "pending_human", ""])
        if topic_index < 16:
            case_id = f"py314-{topic_index + 1:02d}-4"
            question = f"{title} 官方文档是否规定该功能在所有情况下都能自动修复网络故障？"
            cases.append({
                "id": case_id, "question": question, "family_id": slug,
                "type": "unanswerable", "split": split, "answerable": False,
                "expected_keywords": [], "evidence": [], "review_status": "pending_human",
            })
            review_rows.append([case_id, split, slug, question, "no", "", "", "pending_human", ""])
    manifest = {
        "schema_version": 1, "corpus_id": "python-docs-3.14.7-zh-curated",
        "source_version": VERSION, "license": LICENSE,
        "retrieved_at": RETRIEVED_AT, "documents": documents,
    }
    manifest["corpus_hash"] = digest(json.dumps(documents, ensure_ascii=False, sort_keys=True))
    (CORPUS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    dataset = {
        "schema_version": 2, "dataset_id": "python-docs-3.14.7-v2",
        "source_version": VERSION, "review_status": "pending_human",
        "sealed": False, "cases": cases,
    }
    dataset["dataset_hash"] = digest(json.dumps(cases, ensure_ascii=False, sort_keys=True))
    EVAL.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "split", "family_id", "question", "answerable", "evidence_text", "reviewer_notes", "review_status", "reviewer"])
        writer.writerows(review_rows)
    print(json.dumps({"documents": len(documents), "cases": len(cases), "dataset_hash": dataset["dataset_hash"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

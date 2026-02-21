import csv
import json
import logging
import os
import re
import subprocess
from typing import Any

import requests

# ==========================================
# 配置部分 (Configuration)
# ==========================================

class Config:
    # 基础设置
    USE_COUNT = 1000
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CLONE_BASE_DIR = os.path.join(BASE_DIR, 'third_package_source')
    INPUT_FILE = os.path.join(BASE_DIR, 'top-pypi-packages.json')
    OUTPUT_FILE = os.path.join(BASE_DIR, 'pypi_github_languages.csv')
    SUMMARY_FILE = os.path.join(BASE_DIR, 'pypi_github_languages_summary.md')
    
    # 关键字到扩展类型的映射
    KEYWORD_TO_EXTENSION = {
        'PyMODINIT_FUNC': 'C/C++ Extensions',
        'PyInit_': 'C/C++ Extensions',
        'PyModuleDef': 'C/C++ Extensions',
        'PYBIND11_MODULE': 'PyBind11',
        'pybind11/pybind11.h': 'PyBind11',
        'BOOST_PYTHON_MODULE': 'Boost.Python',
        'boost/python.hpp': 'Boost.Python',
        'cpdef': 'Cython',
        'cimport': 'Cython',
        'ctypedef': 'Cython',
        '#[pymodule]': 'Rust',
        '#[pyfunction]': 'Rust',
        '#[pyclass]': 'Rust',
        '%module': 'SWIG',
        '%include': 'SWIG',
        '%typemap': 'SWIG',
        'CDLL(': 'ctypes',
        'WinDLL(': 'ctypes',
        'FFI(': 'CFFI',
        'ffi.cdef(': 'CFFI',    
        'ffi.new(': 'CFFI',
        'ffi.dlopen(': 'CFFI',
    }

# ==========================================
# 日志配置 (Logging)
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 工具函数 (Utility Functions)
# ==========================================

def normalize_github_url(url: str) -> tuple[str, str] | None:
    """标准化GitHub URL并返回(owner, repo)元组"""
    if not url or 'github.com' not in url:
        return None

    if url.endswith('.git'):
        url = url[:-4]

    match = re.search(r'github\.com/([^/]+)/([^/?#]+)', url)
    if match:
        owner, repo = match.groups()
        return owner, repo

    return None

def extract_github_info(project_urls: dict[str, str]) -> tuple[str, str] | None:
    """从project_urls中提取GitHub仓库信息 (owner, repo)"""
    if not project_urls:
        return None

    github_keys = ['Source', 'Code', 'Homepage', 'GitHub', 'Repository']
    for key in github_keys:
        if key in project_urls:
            info = normalize_github_url(project_urls[key])
            if info: return info

    for url in project_urls.values():
        info = normalize_github_url(url)
        if info: return info

    return None


# ==========================================
# 核心逻辑 (Core Logic)
# ==========================================

def clone_or_update_repo(owner: str, repo: str) -> str | None:
    """克隆或更新GitHub仓库到本地"""
    repo_path = os.path.join(Config.CLONE_BASE_DIR, f"{owner}_{repo}")
    if os.path.exists(repo_path):
        logger.info(f"✅ 仓库已存在，跳过克隆: {repo_path}")
        return repo_path

    if not os.path.exists(Config.CLONE_BASE_DIR):
        os.makedirs(Config.CLONE_BASE_DIR)

    repo_url = f"https://github.com/{owner}/{repo}.git"
    logger.info(f"📥 正在克隆仓库: {repo_url}")
    
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, repo_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=300 # 5分钟超时
        )
        return repo_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, 'stderr', str(e))
        logger.error(f"❌ 克隆失败 ({owner}/{repo}): {stderr}")
        return None


def search_keywords_in_local_repo(repo_path: str) -> list[dict[str, Any]]:
    """在本地仓库目录中搜索特征关键字"""
    results: list[dict[str, Any]] = []
    # 维护一个待查找的关键字集合，一旦找到就移除，实现“只搜索没有找到的关键字”
    remaining_keywords: set[str] = set(Config.KEYWORD_TO_EXTENSION.keys())
    found_keywords_global: set[str] = set()
    
    try:
        for root, dirs, files in os.walk(repo_path):
            if '.git' in dirs:
                dirs.remove('.git')
                
            for file in files:
                # 如果所有关键字都已找到，提前结束搜索
                if not remaining_keywords:
                    break
                
                file_path = os.path.join(root, file)
                try:
                    # 限制搜索的文件类型以提高效率，或者保持原样搜索所有文本文件
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # 只搜索当前尚未找到的关键字
                        found_in_this_file = [kw for kw in remaining_keywords if kw in content]
                        for kw in found_in_this_file:
                            found_keywords_global.add(kw)
                            remaining_keywords.remove(kw)
                except Exception as e:
                    logger.warning(f"无法读取文件 {file_path}: {e}")
                    continue
            
            # 如果所有关键字都已找到，提前结束外层循环
            if not remaining_keywords:
                break
    except Exception as e:
        logger.error(f"搜索文件时出错: {e}")
                
    # 将发现的关键字按扩展类型分组
    type_to_found_keywords: dict[str, list[str]] = {}
    for kw in found_keywords_global:
        ext_type = Config.KEYWORD_TO_EXTENSION[kw]
        if ext_type not in type_to_found_keywords:
            type_to_found_keywords[ext_type] = []
        type_to_found_keywords[ext_type].append(kw)
            
    for ext_type, keywords in type_to_found_keywords.items():
        results.append({"type": ext_type, "keywords": keywords})
            
    return results


def process_single_package(package: dict[str, Any]) -> dict[str, Any] | None:
    """处理单个包的逻辑"""
    project_name = package['project']
    download_count = package.get('download_count', 0)
    
    try:
        # 1. 获取 PyPI 信息
        pypi_url = f"https://pypi.org/pypi/{project_name}/json"
        pypi_response = requests.get(pypi_url, timeout=10)
        pypi_response.raise_for_status()
        pypi_data = pypi_response.json()
        project_urls = pypi_data.get('info', {}).get('project_urls', {})

        # 2. 提取 GitHub 信息
        github_info = extract_github_info(project_urls)
        github_url = ""
        extensions: list[dict[str, Any]] = []
        
        if github_info:
            owner, repo = github_info
            github_url = f"https://github.com/{owner}/{repo}"
            
            # 3. 直接克隆并搜索关键字
            repo_path = clone_or_update_repo(owner, repo)
            if repo_path:
                extensions = search_keywords_in_local_repo(repo_path)

        return {
            'project_name': project_name,
            'download_count': download_count,
            'github_url': github_url,
            'pypi_url': f"https://pypi.org/project/{project_name}/",
            'extensions': json.dumps(extensions, ensure_ascii=False)
        }
    except Exception as e:
        logger.error(f"处理包 {project_name} 时发生严重错误: {e}")
        return None


def main() -> None:
    # 加载数据
    try:
        with open(Config.INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        packages = data['rows'][:Config.USE_COUNT]
        logger.info(f"成功加载 {len(packages)} 个包数据")
    except Exception as e:
        logger.error(f"无法加载输入文件: {e}")
        return

    results: list[dict[str, Any]] = []
    for i, pkg in enumerate(packages):
        logger.info(f"[{i+1}/{len(packages)}] 正在处理: {pkg['project']}")
        res = process_single_package(pkg)
        if res:
            results.append(res)
            if json.loads(res['extensions']):
                logger.info(f"✨ 发现扩展模块: {res['extensions']}")

    # 保存结果
    if not results:
        logger.warning("没有收集到任何结果。")
        return

    try:
        fieldnames = ['project_name', 'download_count', 'github_url', 'pypi_url', 'extensions']
        with open(Config.OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"结果已保存到: {Config.OUTPUT_FILE}")
        ext_count = sum(1 for r in results if json.loads(r['extensions']))
        logger.info(f"总计处理: {len(results)} 个项目，其中包含扩展模块: {ext_count} 个")
    except Exception as e:
        logger.error(f"保存 CSV 失败: {e}")
        return

    try:
        total_packages = len(packages)
        github_found = sum(1 for r in results if r.get('github_url'))
        type_to_repo_count: dict[str, int] = {}
        for r in results:
            extensions = json.loads(r['extensions'])
            found_types = {item.get('type') for item in extensions if item.get('type')}
            for ext_type in found_types:
                type_to_repo_count[ext_type] = type_to_repo_count.get(ext_type, 0) + 1

        with open(Config.SUMMARY_FILE, 'w', encoding='utf-8') as mdfile:
            mdfile.write("# PyPI GitHub 扩展类型汇总\n\n")
            mdfile.write("## 总览\n")
            mdfile.write(f"- 总包数: {total_packages}\n")
            mdfile.write(f"- 找到 GitHub 仓库数: {github_found}\n")
            mdfile.write("\n## 各扩展类型覆盖的仓库数\n")
            if type_to_repo_count:
                for ext_type in sorted(type_to_repo_count.keys()):
                    mdfile.write(f"- {ext_type}: {type_to_repo_count[ext_type]}\n")
            else:
                mdfile.write("- 无\n")

        logger.info(f"汇总统计已保存到: {Config.SUMMARY_FILE}")
    except Exception as e:
        logger.error(f"保存汇总统计失败: {e}")

if __name__ == "__main__":
    main()

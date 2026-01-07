import csv
import json
import re
import time
from datetime import datetime
from typing import Dict, Optional, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

use_count = 1000

# GitHub API Token - 请填入你的GitHub Personal Access Token
# 获取方式: https://github.com/settings/tokens
GITHUB_TOKEN = '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'  # 也可以直接填入: GITHUB_TOKEN = 'your_token_here'

# 全局GitHub API会话
github_session: requests.Session | None = None

# 语言到可能扩展类型的映射
LANGUAGE_TO_EXTENSIONS = {
    'C': ['C/C++ Extensions', 'ctypes', 'CFFI', 'SWIG'],
    'C++': ['C/C++ Extensions', 'PyBind11', 'Boost.Python', 'ctypes', 'CFFI', 'SWIG'],
    'Rust': ['Rust'],
    # 'Python': ['Cython', 'ctypes', 'CFFI', 'SWIG'],
    'Cython': ['Cython'],
    'Go': ['ctypes', 'CFFI'],
    'Swift': ['ctypes', 'CFFI'],
    'Java': ['ctypes', 'CFFI', 'SWIG'],
}

# 扩展模块的特征关键字
EXTENSION_PATTERNS = {
    'C/C++ Extensions': [
        'PyModuleDef',  # Python C API
        'PyInit_',      # C扩展模块初始化函数
        'PyObject',     # Python对象
        'PyMethodDef',  # Python方法定义
        'PyArg_ParseTuple',  # 参数解析
        'static PyObject *',  # C扩展函数签名
        'PyModule_Create',  # 创建模块
        '#include <Python.h>',  # Python C头文件
    ],
    'PyBind11': [
        'PYBIND11_MODULE',  # PyBind11模块定义
        'pybind11::',       # PyBind11命名空间
        'm.def(',           # PyBind11函数定义
        'class_',           # PyBind11类绑定
        'py::',             # PyBind11简写
    ],
    'Boost.Python': [
        'BOOST_PYTHON_MODULE',  # Boost.Python模块定义
        'boost::python::',      # Boost.Python命名空间
        'class_<',              # Boost.Python类绑定
        'def(',                 # Boost.Python方法定义
    ],
    'Cython': [
        'cdef ',          # Cython定义
        'cpdef ',         # Cython定义+Python接口
        '.pyx',           # Cython文件扩展名
        'pxd',            # Cython声明文件
        'cimport',        # Cython导入
        'cdef class',     # Cython类定义
    ],
    'Rust': [
        'pyo3::',         # PyO3 Rust绑定
        '#[pyclass]',     # PyO3类装饰器
        '#[pymodule]',    # PyO3模块装饰器
        '#[pyfunction]',  # PyO3函数装饰器
        'maturin',        # Rust Python打包工具
        'pyo3',           # PyO3库
    ],
    'SWIG': [
        '%module ',       # SWIG模块定义
        '%include ',      # SWIG包含
        '%typemap',       # SWIG类型映射
        'SWIGTYPE',       # SWIG类型
    ],
    'ctypes': [
        'ctypes.',        # ctypes库使用
        'CDLL',           # ctypes加载DLL
        'ctypes.CDLL',    # ctypes完整路径
        'WinDLL',         # Windows DLL
        'Structure',      # ctypes结构体
    ],
    'CFFI': [
        'cffi.',          # CFFI库
        'ffi.',           # CFFI FFI对象
        'cdef()',         # CFFI C定义
        'ffi.new()',      # CFFI创建C对象
        'ffi.dlopen()',   # CFFI动态库
    ]
}

def create_github_session():
    """创建配置了重试策略的GitHub API会话"""
    session = requests.Session()
    
    # 重试策略配置 - 只对服务器错误进行重试，速率限制由 safe_github_get 处理
    retry_strategy = Retry(
        total=3,                     # 总重试次数
        status_forcelist=[500, 502, 503, 504],  # 只对服务器错误重试
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # 允许重试的HTTP方法
        backoff_factor=1,           # 重试间隔因子
        raise_on_status=False       # 不抛出状态异常，让我们自己处理
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://api.github.com", adapter)
    
    # 设置默认headers
    session.headers.update({
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'PyPI-GitHub-Language-Analyzer'
    })
    
    if GITHUB_TOKEN:
        session.headers.update({'Authorization': f'token {GITHUB_TOKEN}'})
    
    return session


def safe_github_get(url: str, params: Optional[Dict] = None, max_retries: int = 5) -> requests.Response:
    """
    安全的GitHub API请求包装函数，自动处理速率限制
    
    Args:
        url: 请求URL
        params: 请求参数
        max_retries: 最大重试次数（包括速率限制重试）
    
    Returns:
        requests.Response
    
    Raises:
        requests.exceptions.HTTPError: 如果请求失败且达到最大重试次数
    """
    session = github_session
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = session.get(url, params=params, timeout=10)
            
            # 处理速率限制
            if response.status_code in [403, 429]:
                # 检查是否是速率限制
                if 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
                    reset_time = response.headers.get('X-RateLimit-Reset')
                    if reset_time:
                        reset_timestamp = int(reset_time)
                        current_timestamp = int(time.time())
                        wait_seconds = max(0, reset_timestamp - current_timestamp)
                        
                        # 格式化等待时间显示
                        reset_datetime = datetime.fromtimestamp(reset_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"⏳ GitHub API速率限制，等待 {wait_seconds} 秒 (重置时间: {reset_datetime})")
                        
                        time.sleep(wait_seconds)
                        retry_count += 1
                        continue
                
                # 检查 Retry-After 头部
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait_seconds = int(retry_after)
                        print(f"⏳ GitHub API请求过多，等待 {wait_seconds} 秒后重试")
                        time.sleep(wait_seconds)
                        retry_count += 1
                        continue
                    except ValueError:
                        pass
                
                # 如果没有明确的等待时间，使用默认等待
                print(f"⚠️  GitHub API返回 {response.status_code}，等待 60 秒后重试")
                time.sleep(60)
                retry_count += 1
                continue
            
            # 处理其他错误状态
            if response.status_code != 200:
                error = handle_response_errors(response)
                print(f"❌ 请求失败: {error} - {url}")
                response.raise_for_status()
            
            return response
            
        except requests.exceptions.Timeout:
            print(f"请求超时: {url}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"等待 5 秒后重试 ({retry_count}/{max_retries})")
                time.sleep(5)
                continue
            raise
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {str(e)} - {url}")
            raise
    
    raise requests.exceptions.HTTPError(f"达到最大重试次数 ({max_retries})，放弃请求: {url}")


def handle_response_errors(response: requests.Response) -> str | None:
    """处理API响应错误，返回错误信息或None"""
    if response.status_code == 200:
        return None
    
    if response.status_code == 403:
        if 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
            reset_time = response.headers.get('X-RateLimit-Reset', 0)
            return f"API速率限制，重置时间: {reset_time}"
        else:
            return "访问被禁止，可能需要token"
    elif response.status_code == 404:
        return "资源未找到"
    elif response.status_code >= 500:
        return f"服务器错误: {response.status_code}"
    else:
        return f"HTTP错误: {response.status_code}"


def load_pypi_data(json_file: str) -> List[Dict]:
    """加载PyPI数据JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 只取前n个最流行的库
    return data['rows'][:use_count]

def extract_github_info(project_urls: Dict[str, str]) -> Optional[tuple]:
    """从project_urls中提取GitHub仓库信息 (owner, repo)"""
    if not project_urls:
        return None

    # 常见的GitHub相关key名称
    github_keys = ['Source', 'Code', 'Homepage', 'GitHub', 'Repository']

    # 首先检查常见的key
    for key in github_keys:
        if key in project_urls:
            url = project_urls[key]
            if 'github.com' in url:
                return normalize_github_url(url)

    # 检查所有URL，寻找GitHub链接
    for url in project_urls.values():
        if url and 'github.com' in url:
            return normalize_github_url(url)

    return None

def normalize_github_url(url: str) -> Optional[tuple]:
    """标准化GitHub URL并返回(owner, repo)元组"""
    if not url or 'github.com' not in url:
        return None

    # 移除.git后缀 - 使用 endswith 而不是 rstrip
    if url.endswith('.git'):
        url = url[:-4]

    # 确保是正确的GitHub格式
    # 提取 owner/repo 部分
    match = re.search(r'github\.com/([^/]+)/([^/?#]+)', url)
    if match:
        owner, repo = match.groups()
        return owner, repo

    return None

def build_github_urls(owner: str, repo: str) -> Dict[str, str]:
    """根据owner和repo构建各种GitHub URL"""
    return {
        'api_url': f"https://api.github.com/repos/{owner}/{repo}",
        'web_url': f"https://github.com/{owner}/{repo}",
        'repo_name': f"{owner}/{repo}"
    }

def get_github_languages(owner: str, repo: str) -> Optional[Dict[str, int]]:
    """通过GitHub API获取仓库的语言信息"""
    if not owner or not repo:
        return None
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    
    # 使用 safe_github_get 自动处理速率限制
    response = safe_github_get(api_url)

    return response.json()

def format_languages(languages: Dict[str, int]) -> str:
    """格式化语言信息为JSON数组字符串"""
    if not languages:
        return "[]"

    # 计算总字节数
    total = sum(languages.values())

    # 创建语言数组，每个元素包含语言名和百分比
    lang_array = []
    for lang, bytes_count in languages.items():
        percentage = round((bytes_count / total) * 100, 2)
        lang_array.append({
            "language": lang,
            "percentage": percentage
        })

    # 按百分比降序排序
    lang_array.sort(key=lambda x: x['percentage'], reverse=True)

    # 返回JSON数组格式的字符串
    return json.dumps(lang_array, ensure_ascii=False)

def search_code_in_repo(owner: str, repo: str, languages: Dict[str, int]) -> List[str]:
    """在GitHub仓库中搜索代码模式"""
    if not GITHUB_TOKEN:
        print("跳过代码搜索: 未设置GitHub Token")
        return []

    extensions_found = []

    # 使用传入的语言信息
    lang_data = {lang: bytes_count for lang, bytes_count in languages.items()}

    # 确定需要搜索的扩展类型
    ext_types_to_search = set()
    for lang in lang_data.keys():
        if lang in LANGUAGE_TO_EXTENSIONS:
            ext_types_to_search.update(LANGUAGE_TO_EXTENSIONS[lang])

    repo_name = f"{owner}/{repo}"
    print(f"搜索 {repo_name} 的代码模式: {ext_types_to_search}")

    # 如果没有相关语言，返回空
    if not ext_types_to_search:
        return extensions_found

    # 只搜索相关的扩展类型
    for ext_type in ext_types_to_search:
        found_patterns = []

        # 获取该扩展类型的模式
        patterns = EXTENSION_PATTERNS.get(ext_type, [])
        if not patterns:
            continue

        # 搜索每个模式（限制搜索次数以避免API限制）
        for pattern in patterns[:2]:  # 每种类型最多搜索2个模式
            # 构建搜索查询
            query = f'repo:{repo_name} "{pattern}"'

            # 使用params参数正确处理URL编码
            search_params = {
                'q': query,
                'per_page': 1  # 只需要知道是否存在，不需要具体结果
            }

            # 使用 safe_github_get 自动处理速率限制
            search_response = safe_github_get(
                'https://api.github.com/search/code',
                params=search_params
            )

            if search_response.json().get('total_count', 0) > 0:
                found_patterns.append(pattern)

        if found_patterns:
            extensions_found.append({
                "type": ext_type,
                "patterns": found_patterns
            })

    return extensions_found

def process_packages(packages: List[Dict]) -> List[Dict]:
    """处理所有包数据"""
    results = []

    for i, package in enumerate(packages):
        print(f"处理进度: {i+1}/{len(packages)} - {package['project']}")

        # 获取PyPI信息
        project_name = package['project']
        download_count = package.get('download_count', 0)

        # 获取PyPI详细信息
        pypi_url = f"https://pypi.org/pypi/{project_name}/json"
        pypi_response = requests.get(pypi_url, timeout=10)
        pypi_response.raise_for_status()

        pypi_data = pypi_response.json()
        project_urls = pypi_data.get('info', {}).get('project_urls', {})

        # 提取GitHub信息 (owner, repo)
        github_info = extract_github_info(project_urls)
        github_url = ""
        languages_str = ""
        extensions = []
        
        if github_info:
            owner, repo = github_info
            urls = build_github_urls(owner, repo)
            github_url = urls['web_url']
            
            # 获取语言信息
            languages = get_github_languages(owner, repo)
            languages_str = format_languages(languages) if languages else ""
            
            # 搜索扩展模块
            if languages:
                extensions = search_code_in_repo(owner, repo, languages)

        # 记录结果
        result = {
            'project_name': project_name,
            'download_count': download_count,
            'github_url': github_url,
            'pypi_url': f"https://pypi.org/project/{project_name}/",
            'languages': languages_str,
            'extensions': json.dumps(extensions, ensure_ascii=False)
        }

        results.append(result)

        print(f'extensions: {extensions}\n')

    return results

def save_to_csv(results: List[Dict], output_file: str):
    """保存结果到CSV文件"""
    fieldnames = ['project_name', 'download_count', 'github_url', 'pypi_url', 'languages', 'extensions']

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存到: {output_file}")
    print(f"总计处理: {len(results)} 个项目")

    # 统计扩展信息
    ext_count = 0
    for result in results:
        extensions = json.loads(result['extensions'])
        if extensions:
            ext_count += 1

    print(f"包含扩展模块的项目: {ext_count} 个")

def main():
    input_file = 'top-pypi-packages.json'
    output_file = 'pypi_github_languages.csv'

    # 初始化GitHub API会话
    global github_session
    github_session = create_github_session()
    
    print("开始加载PyPI数据...")
    packages = load_pypi_data(input_file)
    print(f"成功加载 {len(packages)} 个包数据")

    print("\n开始处理包信息...")
    results = process_packages(packages)

    print("\n保存结果...")
    save_to_csv(results, output_file)

if __name__ == "__main__":
    main()

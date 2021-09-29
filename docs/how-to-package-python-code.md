## 如何将自己的Python包发布到PyPI

1. 编写 setup.py
2. 更新 setuptools，因为setuptools>= 38.6.0 才能使用新的元数据生成发布包
```python
pip install -U setuptools
```
3. 用 twine上传分发包，并且只有 twine> = 1.11.0 才能将元数据正确发送到 Pypi上
```python
pip install -U twine
```

- 打包项目并上传

运行`python setup.py check`检查setup.py是否有错误，如果没报错误，则进行下一步

注册[PyPI]()帐号，注册完成之后，在本机（Linux或者Mac）创建~/.pypirc文件，文件内容如下
```shell
[distutils]
index-servers=pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = xxxx  # pypi登录用户名
password = xxxx  # pypi登录密码
```
4. 生成源码分发包
```python
python setup.py sdist
```

运行该命令之后，会生成一个haipproxy.egg-info文件夹，可以查看其中的SOURCES.txt文件，以确定是否所有需要的内容都已经被包括在待发布的包中

5. 上传分发包
```python
twine upload dist/* # 也可以单独指定 dist 文件夹中的某个版本的发布包
```
上传成功之后，便可以使用pip install package_name来安装和使用发布的包了

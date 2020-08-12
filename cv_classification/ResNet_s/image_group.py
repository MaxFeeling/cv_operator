# -*- coding: utf-8 -*-

import logging
from uuid import uuid4

from sage._4pd.datasource.image_set import ImageSetDataSource
from sage._4pd.datasource.image_group import ImageGroupDataSource
from sage.core.api.v1.compute.compute import ComputeService
from sage.core.api.v1.data.data import DataService
from sage.context import SageContext, get_current_context
from sage.util.env import in_sage_notebook_env
from sage.core.api.v1.compute.types import ImageGroupConvertRequest
from sage.core.api.v1.data.types import Protocol


logging.getLogger('sage').setLevel(logging.INFO)

if not in_sage_notebook_env():
    context = SageContext('http://172.27.128.150:40121',
                          access_key='e5f1fc70-7d1d-4f60-9bd6-fd949078adad',
                          workspace_id=1)
    context.start()

context = get_current_context()
context.config['defaults.sage.context.ignore_version_check'] = True
context.debug_requests_on()

# 1. 本地上传到hdfs
local_path='D:\ResNet_s\mnist_png.zip'
image_set_url = ComputeService.upload_file(local_path)

# 2. import imageset
image_set_prn="example/image-%s.image-set" % (uuid4().hex)
image_set_source = ImageSetDataSource(
    url=image_set_url,
    protocol=Protocol.HDFS,
    prn=image_set_prn
)
output = image_set_source()
output.submit()

# 3. imageset 引入成功后，获取初始标注csv文件
image_set = DataService.get_image_set(image_set_prn)
urls =  image_set.to_dict().get('supplementsUrls')
csv_url = ""
for url in urls:
    if url.endswith("zip"):
        csv_url = 'hdfs://' + url[0: url.find('COMPRESSED_FILE')] + 'META-INFO/labels.csv'


# import imagegroup 标注信息
image_group_prn = 'example/label.image-group'
image_group_source = ImageGroupDataSource(
    url=csv_url,
    protocol=Protocol.HDFS,
    prn=image_group_prn
)
output = image_group_source()
output.submit()

# 4. imagegroup 引入成功后，标注信息更新，重新生成新的标注csv文件
request = ImageGroupConvertRequest.from_props(prn=image_group_prn)
response = ComputeService.export_image_group(request)
job = ComputeService.get_jobs(response.job_prn)

# 获取标注图片集
image_group = DataService.get_image_group(image_group_prn)

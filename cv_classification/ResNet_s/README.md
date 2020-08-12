USAGE:
    
    Step 1. 添加当前项目的setup.py的绝对路径到~/.algo_cli/config.ini的[scan_paths]配置中

    Step 2. unzip 当前目录下的示例数据,执行: unzip mnist_png.zip。

    Step 3. 运行执行命令:  algo_cli run resnet_series --model_dir ./models  --input_data_source  ./mnist_png/labels.csv  --batch_size 64 

    Step 4. 运行执行命令:  algo_cli run resnet_test --model_dir ./models  --inputs  ./models/saved_model,./mnist_png/labels.csv  --batch_size 64 

    Step 5. algo_cli deploy resnet_series 到先知平台，可以在界面拖拽图片集数据运行。 运行前需将内存更改为12个g。

    Step 6. algo_cli deploy resnet_test到先知平台，可在界面拖拽图片集数据运行。运行前需更改内存为12个g
    

   
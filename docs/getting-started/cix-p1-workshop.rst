CIX P1 Workshop：预编译环境准备
================================================================

本页使用 FlagTree-CPU、FlagGems 和 vLLM-plugin-FL 的固定版本预编译包，
准备 MiniCPM-V-4.7-1B W8A8 的 FlagOS 开发环境，无需现场编译组件。

主机条件
----------------------------------------------------------------

在 CIX P1 的 Linux aarch64 主机上使用 ``baai`` 账号，HOME 为
``/home/baai``。系统可见内存至少 15 GiB，首次准备建议预留 25 GiB
磁盘空间。主机需要 Docker、Python 3.9+、python3-venv、Git、curl、tar、
gzip、sha256sum，以及 ``sudo -n docker`` 权限和 Harbor 拉取权限。
本教程绑定八个 Cortex-A720：``0,1,6,7,8,9,10,11``；核心编号不同时
需要先调整并验证亲和性配置。网络需可访问 GitHub、ModelScope、Python
包镜像与 Harbor；受限网络先配置终端代理。

下载并准备材料
----------------------------------------------------------------

在 P1 主机执行以下两条命令。也可直接下载
:download:`准备脚本 <scripts/prepare_workshop.sh>`。

.. code-block:: bash

   curl -fLO https://raw.githubusercontent.com/kevinzs2048/flagtree-cpu/triton_v3.7.x/docs/getting-started/scripts/prepare_workshop.sh
   bash prepare_workshop.sh

脚本在 ``/home/baai/workshop1018`` 下载操作脚本、模型、预编译 wheel
和性能对照所需的 plugin 源码，并从 Harbor 拉取固定镜像，最后校验材料。
镜像地址来自随包的 ``scripts/release.env``，不需要手工填写长镜像名。
登录 Harbor 时按提示输入获授权账号。

看到 ``Materials ready`` 后继续下文。网络中断后可重新运行脚本；它只允许
继续准备自己创建且尚未安装的目录，不覆盖其他已有目录。
已有预置环境直接从下面的“检查材料”开始。

检查材料并创建开发容器
----------------------------------------------------------------

以下命令在 P1 主机执行：

.. code-block:: bash

   export WORK_ROOT=/home/baai/workshop1018
   source "$WORK_ROOT/scripts/activate_host.env"
   bash "$WORK_ROOT/scripts/check_preloaded.sh"

看到 ``PRELOADED_CHECK_PASS`` 后创建容器：

.. code-block:: bash

   CONTAINER=flagos-workshop1018 bash "$WORK_ROOT/scripts/init_container.sh"
   sudo docker exec -it flagos-workshop1018 /bin/bash

``flagos-workshop1018`` 是此处创建的容器名称，不是 Harbor 镜像名称。
模型目录只读挂载到 ``/model``，工作目录挂载到 ``/work``。

安装预编译组件（容器内）
----------------------------------------------------------------

进入开发容器后依次执行；环境准备与安装只需完成一次：

.. code-block:: bash

   /opt/flagos-runtime/bin/flagos-python --profile 1b /work/scripts/prepare_env.py
   bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py preflight /work
   bash /work/scripts/dev_python.sh -m pip install --no-index --no-deps --no-compile \
     --target /work/envs/dev/lib/python3.11/site-packages /work/prebuilt/wheels/*.whl
   bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py installed /work

看到 ``PREBUILT_IMPORT_AND_RAW_LOAD_PASS`` 表示安装检查通过。
FlagTree-CPU 的 wheel 包名为 ``triton``；基础镜像已经包含 Python、PyTorch、
vLLM 0.24 CPU 等依赖。预编译组件安装后，首次推理仍可能需要数分钟 JIT。
准备脚本不会自动启动推理服务；后续验证和服务操作使用工作目录内的脚本。

发布材料说明
----------------------------------------------------------------

本页与准备脚本维护在 FlagTree-CPU 文档中。脚本继续引用已发布的 Workshop
工具包、FlagGems wheel 包、ModelScope 模型和 Harbor 镜像，不复制或改写这些
组件的发布文件。模型、镜像、wheel 与基线源码均按随包清单校验。

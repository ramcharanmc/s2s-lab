✓ Initialized. View run at 
https://modal.com/apps/ramcharanmc/main/ap-SmPdw14AmM4a7XR0BiNNnd
Building image im-a1s1CAsGuNtCH2gEq34YaN

=> Step 0: FROM base

=> Step 1: RUN python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
Looking in indexes: https://download.pytorch.org/whl/cu126
Requirement already satisfied: torch==2.14.0 in /usr/local/lib/python3.12/site-packages (2.14.0+cu126)
Collecting torchvision==0.29.0
  Downloading https://download-r2.pytorch.org/whl/cu126/torchvision-0.29.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (5.7 kB)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (3.32.3)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (4.16.0)
Requirement already satisfied: setuptools>=77.0.3 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (78.1.0)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (2026.7.0)
Requirement already satisfied: cuda-toolkit==12.6.3 in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (12.6.3)
Requirement already satisfied: cuda-bindings<13,>=12.9.4 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (12.9.7)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (9.10.2.21)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.29.3 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (2.29.3)
Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (3.4.5)
Requirement already satisfied: nvidia-nvjitlink-cu12<13,>=12.6.85 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (12.9.86)
Requirement already satisfied: triton~=3.8.0 in /usr/local/lib/python3.12/site-packages (from torch==2.14.0) (3.8.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/site-packages (from torchvision==0.29.0) (2.2.6)
Collecting pillow!=8.3.*,>=5.3.0 (from torchvision==0.29.0)
  Downloading pillow-12.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.1 kB)
Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (12.6.4.1)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (12.6.77)
Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (11.3.0.4)
Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (1.11.1.6)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (12.6.80)
Requirement already satisfied: nvidia-curand-cu12==10.3.7.77.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (10.3.7.77)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (11.7.1.2)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (12.5.4.2)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.85.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (12.6.85)
Requirement already satisfied: nvidia-nvtx-cu12==12.6.77.* in /usr/local/lib/python3.12/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3; platform_system == "Linux"->torch==2.14.0) (12.6.77)
Requirement already satisfied: cuda-pathfinder~=1.1 in /usr/local/lib/python3.12/site-packages (from cuda-bindings<13,>=12.9.4->torch==2.14.0) (1.6.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/site-packages (from sympy>=1.13.3->torch==2.14.0) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/site-packages (from jinja2->torch==2.14.0) (3.0.3)
Downloading https://download-r2.pytorch.org/whl/cu126/torchvision-0.29.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl (7.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.2/7.2 MB 112.1 MB/s eta 0:00:00
Downloading pillow-12.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 95.1 MB/s eta 0:00:00
Installing collected packages: pillow, torchvision
Successfully installed pillow-12.3.0 torchvision-0.29.0+cu126
Saving image...
Image saved, took 538.83ms

Built image im-a1s1CAsGuNtCH2gEq34YaN in 6.02s


Building image im-XHULW6PZR6RwPwFwUq8CAn

=> Step 0: FROM base

=> Step 1: RUN python -m pip install pillow==11.3.0
Looking in indexes: http://pypi-mirror.modal.local:5555/simple
Collecting pillow==11.3.0
  Downloading http://pypi-mirror.modal.local:5555/simple/pillow/pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
Downloading http://pypi-mirror.modal.local:5555/simple/pillow/pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 6.3 MB/s eta 0:00:00
Installing collected packages: pillow
  Attempting uninstall: pillow
    Found existing installation: pillow 12.3.0
    Uninstalling pillow-12.3.0:
      Successfully uninstalled pillow-12.3.0
Successfully installed pillow-11.3.0

[notice] A new release of pip is available: 23.3.2 -> 26.2.1
[notice] To update, run: pip install --upgrade pip
Saving image...
Image saved, took 874.88ms

Built image im-XHULW6PZR6RwPwFwUq8CAn in 9.74s


Building image im-5brGOZd92uODnlCr0MFCus

=> Step 0: FROM base

=> Step 1: RUN apt-get update
Get:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease [1578 B]
Get:2 http://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]
Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]
Get:4 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  Packages [3037 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [127 kB]
Get:7 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [4354 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy/universe amd64 Packages [17.5 MB]
Get:9 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [7819 kB]
Get:10 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [84.6 kB]
Get:11 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1318 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages [1792 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy/multiverse amd64 Packages [266 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy/restricted amd64 Packages [164 kB]
Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [8115 kB]
Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [4692 kB]
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [93.2 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1622 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [35.6 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [82.9 kB]
Fetched 51.6 MB in 4s (12.7 MB/s)
Reading package lists...

=> Step 2: RUN apt-get install -y gcc
Reading package lists...
Building dependency tree...
Reading state information...
The following additional packages will be installed:
  binutils binutils-common binutils-x86-64-linux-gnu cpp cpp-11
  fontconfig-config fonts-dejavu-core gcc-11 gcc-11-base gcc-12-base libasan6
  libatomic1 libbinutils libbrotli1 libbsd0 libc-dev-bin libc-devtools libc6
  libc6-dev libcc1-0 libcrypt-dev libctf-nobfd0 libctf0 libdeflate0 libexpat1
  libfontconfig1 libfreetype6 libgcc-11-dev libgcc-s1 libgd3 libgomp1 libisl23
  libitm1 libjbig0 libjpeg-turbo8 libjpeg8 liblsan0 libmd0 libmpc3 libmpfr6
  libnsl-dev libpng16-16 libquadmath0 libstdc++6 libtiff5 libtirpc-dev
  libtsan0 libubsan1 libwebp7 libx11-6 libx11-data libxau6 libxcb1 libxdmcp6
  libxpm4 linux-libc-dev manpages manpages-dev rpcsvc-proto ucf
Suggested packages:
  binutils-doc cpp-doc gcc-11-locales gcc-multilib make autoconf automake
  libtool flex bison gdb gcc-doc gcc-11-multilib gcc-11-doc glibc-doc locales
  libgd-tools man-browser
Recommended packages:
  libnss-nis libnss-nisplus
The following NEW packages will be installed:
  binutils binutils-common binutils-x86-64-linux-gnu cpp cpp-11
  fontconfig-config fonts-dejavu-core gcc gcc-11 gcc-11-base libasan6
  libatomic1 libbinutils libbrotli1 libbsd0 libc-dev-bin libc-devtools
  libc6-dev libcc1-0 libcrypt-dev libctf-nobfd0 libctf0 libdeflate0 libexpat1
  libfontconfig1 libfreetype6 libgcc-11-dev libgd3 libgomp1 libisl23 libitm1
  libjbig0 libjpeg-turbo8 libjpeg8 liblsan0 libmd0 libmpc3 libmpfr6 libnsl-dev
  libpng16-16 libquadmath0 libtiff5 libtirpc-dev libtsan0 libubsan1 libwebp7
  libx11-6 libx11-data libxau6 libxcb1 libxdmcp6 libxpm4 linux-libc-dev
  manpages manpages-dev rpcsvc-proto ucf
The following packages will be upgraded:
  gcc-12-base libc6 libgcc-s1 libstdc++6
4 upgraded, 57 newly installed, 0 to remove and 62 not upgraded.
Need to get 61.3 MB of archives.
After this operation, 179 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libc6 amd64 2.35-0ubuntu3.15 [3234 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 gcc-12-base amd64 12.3.0-1ubuntu1~22.04.3 [216 kB]
Get:3 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libstdc++6 amd64 12.3.0-1ubuntu1~22.04.3 [699 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgcc-s1 amd64 12.3.0-1ubuntu1~22.04.3 [53.9 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy/main amd64 libmd0 amd64 1.0.4-1build1 [23.0 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy/main amd64 libbsd0 amd64 0.11.5-1 [44.8 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libexpat1 amd64 2.4.7-1ubuntu0.8 [95.6 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy/main amd64 ucf all 3.0043 [56.1 kB]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpng16-16 amd64 1.6.37-3ubuntu0.6 [192 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxau6 amd64 1:1.0.9-1build5 [7634 B]
Get:11 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxdmcp6 amd64 1:1.1.3-0ubuntu5 [10.9 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxcb1 amd64 1.14-3ubuntu3 [49.0 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libx11-data all 2:1.7.5-1ubuntu0.3 [120 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libx11-6 amd64 2:1.7.5-1ubuntu0.3 [667 kB]
Get:15 http://archive.ubuntu.com/ubuntu jammy/main amd64 manpages all 5.10-1ubuntu1 [1375 kB]
Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-common amd64 2.38-4ubuntu2.12 [223 kB]
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libbinutils amd64 2.38-4ubuntu2.12 [663 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf-nobfd0 amd64 2.38-4ubuntu2.12 [108 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf0 amd64 2.38-4ubuntu2.12 [103 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-x86-64-linux-gnu amd64 2.38-4ubuntu2.12 [2324 kB]
Get:21 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils amd64 2.38-4ubuntu2.12 [3184 B]
Get:22 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 gcc-11-base amd64 11.4.0-1ubuntu1~22.04.3 [216 kB]
Get:23 http://archive.ubuntu.com/ubuntu jammy/main amd64 libisl23 amd64 0.24-2build1 [727 kB]
Get:24 http://archive.ubuntu.com/ubuntu jammy/main amd64 libmpfr6 amd64 4.1.0-3build3 [1425 kB]
Get:25 http://archive.ubuntu.com/ubuntu jammy/main amd64 libmpc3 amd64 1.2.1-2build1 [46.9 kB]
Get:26 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 cpp-11 amd64 11.4.0-1ubuntu1~22.04.3 [10.0 MB]
Get:27 http://archive.ubuntu.com/ubuntu jammy/main amd64 cpp amd64 4:11.2.0-1ubuntu1 [27.7 kB]
Get:28 http://archive.ubuntu.com/ubuntu jammy/main amd64 fonts-dejavu-core all 2.37-2build1 [1041 kB]
Get:29 http://archive.ubuntu.com/ubuntu jammy/main amd64 fontconfig-config all 2.13.1-4.2ubuntu5 [29.1 kB]
Get:30 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libcc1-0 amd64 12.3.0-1ubuntu1~22.04.3 [48.3 kB]
Get:31 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgomp1 amd64 12.3.0-1ubuntu1~22.04.3 [127 kB]
Get:32 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libitm1 amd64 12.3.0-1ubuntu1~22.04.3 [30.2 kB]
Get:33 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libatomic1 amd64 12.3.0-1ubuntu1~22.04.3 [10.5 kB]
Get:34 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libasan6 amd64 11.4.0-1ubuntu1~22.04.3 [2283 kB]
Get:35 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 liblsan0 amd64 12.3.0-1ubuntu1~22.04.3 [1069 kB]
Get:36 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libtsan0 amd64 11.4.0-1ubuntu1~22.04.3 [2260 kB]
Get:37 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libubsan1 amd64 12.3.0-1ubuntu1~22.04.3 [976 kB]
Get:38 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libquadmath0 amd64 12.3.0-1ubuntu1~22.04.3 [154 kB]
Get:39 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgcc-11-dev amd64 11.4.0-1ubuntu1~22.04.3 [2517 kB]
Get:40 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 gcc-11 amd64 11.4.0-1ubuntu1~22.04.3 [20.1 MB]
Get:41 http://archive.ubuntu.com/ubuntu jammy/main amd64 gcc amd64 4:11.2.0-1ubuntu1 [5112 B]
Get:42 http://archive.ubuntu.com/ubuntu jammy/main amd64 libbrotli1 amd64 1.0.9-2build6 [315 kB]
Get:43 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libc-dev-bin amd64 2.35-0ubuntu3.15 [20.3 kB]
Get:44 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libfreetype6 amd64 2.11.1+dfsg-1ubuntu0.3 [388 kB]
Get:45 http://archive.ubuntu.com/ubuntu jammy/main amd64 libfontconfig1 amd64 2.13.1-4.2ubuntu5 [131 kB]
Get:46 http://archive.ubuntu.com/ubuntu jammy/main amd64 libjpeg-turbo8 amd64 2.1.2-0ubuntu1 [134 kB]
Get:47 http://archive.ubuntu.com/ubuntu jammy/main amd64 libjpeg8 amd64 8c-2ubuntu10 [2264 B]
Get:48 http://archive.ubuntu.com/ubuntu jammy/main amd64 libdeflate0 amd64 1.10-2 [70.9 kB]
Get:49 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libjbig0 amd64 2.1-3.1ubuntu0.22.04.1 [29.2 kB]
Get:50 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libwebp7 amd64 1.2.2-2ubuntu0.22.04.2 [206 kB]
Get:51 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libtiff5 amd64 4.3.0-6ubuntu0.13 [185 kB]
Get:52 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libxpm4 amd64 1:3.5.12-1ubuntu0.22.04.3 [36.5 kB]
Get:53 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgd3 amd64 2.3.0-2ubuntu2.3 [129 kB]
Get:54 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libc-devtools amd64 2.35-0ubuntu3.15 [29.0 kB]
Get:55 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 linux-libc-dev amd64 5.15.0-194.204 [1344 kB]
Get:56 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcrypt-dev amd64 1:4.4.27-1 [112 kB]
Get:57 http://archive.ubuntu.com/ubuntu jammy/main amd64 rpcsvc-proto amd64 1.4.2-0ubuntu6 [68.5 kB]
Get:58 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libtirpc-dev amd64 1.3.2-2ubuntu0.1 [192 kB]
Get:59 http://archive.ubuntu.com/ubuntu jammy/main amd64 libnsl-dev amd64 1.3.0-2build2 [71.3 kB]
Get:60 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libc6-dev amd64 2.35-0ubuntu3.15 [2100 kB]
Get:61 http://archive.ubuntu.com/ubuntu jammy/main amd64 manpages-dev all 5.10-1ubuntu1 [2309 kB]
debconf: delaying package configuration, since apt-utils is not installed
Fetched 61.3 MB in 5s (13.0 MB/s)
(Reading database ... 
[1A(Reading database ... 5%
[1A(Reading database ... 10%
[1A(Reading database ... 15%
[1A(Reading database ... 20%
[1A(Reading database ... 25%
[1A(Reading database ... 30%
[1A(Reading database ... 35%(Reading database ... 40%
[1A(Reading database ... 45%(Reading database ... 50%(Reading database ... 55%(Reading database ... 60%(Reading database ... 65%(Reading database ... 70%(Reading database ... 75%(Reading database ... 80%(Reading database ... 85%(Reading database ... 90%(Reading database ... 95%(Reading database ... 100%(Reading database ... 5341 files and directories currently installed.)
Preparing to unpack .../libc6_2.35-0ubuntu3.15_amd64.deb ...
debconf: unable to initialize frontend: Dialog
debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78.)
debconf: falling back to frontend: Readline
debconf: unable to initialize frontend: Readline
debconf: (Can't locate Term/ReadLine.pm in @INC (you may need to install the Term::ReadLine module) (@INC contains: /etc/perl /usr/local/lib/x86_64-linux-gnu/perl/5.34.0 /usr/local/share/perl/5.34.0 /usr/lib/x86_64-linux-gnu/perl5/5.34 /usr/share/perl5 /usr/lib/x86_64-linux-gnu/perl-base /usr/lib/x86_64-linux-gnu/perl/5.34 /usr/share/perl/5.34 /usr/local/lib/site_perl) at /usr/share/perl5/Debconf/FrontEnd/Readline.pm line 7.)
debconf: falling back to frontend: Teletype
Unpacking libc6:amd64 (2.35-0ubuntu3.15) over (2.35-0ubuntu3.8) ...
Setting up libc6:amd64 (2.35-0ubuntu3.15) ...
debconf: unable to initialize frontend: Dialog
debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78.)
debconf: falling back to frontend: Readline
debconf: unable to initialize frontend: Readline
debconf: (Can't locate Term/ReadLine.pm in @INC (you may need to install the Term::ReadLine module) (@INC contains: /etc/perl /usr/local/lib/x86_64-linux-gnu/perl/5.34.0 /usr/local/share/perl/5.34.0 /usr/lib/x86_64-linux-gnu/perl5/5.34 /usr/share/perl5 /usr/lib/x86_64-linux-gnu/perl-base /usr/lib/x86_64-linux-gnu/perl/5.34 /usr/share/perl/5.34 /usr/local/lib/site_perl) at /usr/share/perl5/Debconf/FrontEnd/Readline.pm line 7.)
debconf: falling back to frontend: Teletype
(Reading database ... 
[1A(Reading database ... 5%
[1A(Reading database ... 10%
[1A(Reading database ... 15%
[1A(Reading database ... 20%
[1A(Reading database ... 25%
[1A(Reading database ... 30%
[1A(Reading database ... 35%(Reading database ... 40%
[1A(Reading database ... 45%(Reading database ... 50%(Reading database ... 55%(Reading database ... 60%
[1A(Reading database ... 65%(Reading database ... 70%(Reading database ... 75%(Reading database ... 80%
[1A(Reading database ... 85%(Reading database ... 90%(Reading database ... 95%(Reading database ... 100%
[1A(Reading database ... 5341 files and directories currently installed.)
Preparing to unpack .../gcc-12-base_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking gcc-12-base:amd64 (12.3.0-1ubuntu1~22.04.3) over (12.3.0-1ubuntu1~22.04) ...
Setting up gcc-12-base:amd64 (12.3.0-1ubuntu1~22.04.3) ...
(Reading database ... 
[1A(Reading database ... 5%
[1A(Reading database ... 10%
[1A(Reading database ... 15%
[1A(Reading database ... 20%
[1A(Reading database ... 25%
[1A(Reading database ... 30%
[1A(Reading database ... 35%(Reading database ... 40%(Reading database ... 45%
[1A(Reading database ... 50%(Reading database ... 55%(Reading database ... 60%(Reading database ... 65%(Reading database ... 70%(Reading database ... 75%(Reading database ... 80%(Reading database ... 85%(Reading database ... 90%(Reading database ... 95%(Reading database ... 100%(Reading database ... 5341 files and directories currently installed.)
Preparing to unpack .../libstdc++6_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libstdc++6:amd64 (12.3.0-1ubuntu1~22.04.3) over (12.3.0-1ubuntu1~22.04) ...
Setting up libstdc++6:amd64 (12.3.0-1ubuntu1~22.04.3) ...
(Reading database ... 
[1A(Reading database ... 5%
[1A(Reading database ... 10%
[1A(Reading database ... 15%
[1A(Reading database ... 20%
[1A(Reading database ... 25%
[1A(Reading database ... 30%
[1A(Reading database ... 35%(Reading database ... 40%(Reading database ... 45%(Reading database ... 50%
[1A(Reading database ... 55%(Reading database ... 60%(Reading database ... 65%(Reading database ... 70%(Reading database ... 75%(Reading database ... 80%(Reading database ... 85%
[1A(Reading database ... 90%(Reading database ... 95%(Reading database ... 100%(Reading database ... 5341 files and directories currently installed.)
Preparing to unpack .../libgcc-s1_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libgcc-s1:amd64 (12.3.0-1ubuntu1~22.04.3) over (12.3.0-1ubuntu1~22.04) ...
Setting up libgcc-s1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libmd0:amd64.
(Reading database ... 
[1A(Reading database ... 5%
[1A(Reading database ... 10%
[1A(Reading database ... 15%
[1A(Reading database ... 20%
[1A(Reading database ... 25%
[1A(Reading database ... 30%
[1A(Reading database ... 35%(Reading database ... 40%(Reading database ... 45%(Reading database ... 50%
[1A(Reading database ... 55%(Reading database ... 60%(Reading database ... 65%(Reading database ... 70%(Reading database ... 75%(Reading database ... 80%(Reading database ... 85%
[1A(Reading database ... 90%(Reading database ... 95%(Reading database ... 100%(Reading database ... 5341 files and directories currently installed.)
Preparing to unpack .../00-libmd0_1.0.4-1build1_amd64.deb ...
Unpacking libmd0:amd64 (1.0.4-1build1) ...
Selecting previously unselected package libbsd0:amd64.
Preparing to unpack .../01-libbsd0_0.11.5-1_amd64.deb ...
Unpacking libbsd0:amd64 (0.11.5-1) ...
Selecting previously unselected package libexpat1:amd64.
Preparing to unpack .../02-libexpat1_2.4.7-1ubuntu0.8_amd64.deb ...
Unpacking libexpat1:amd64 (2.4.7-1ubuntu0.8) ...
Selecting previously unselected package ucf.
Preparing to unpack .../03-ucf_3.0043_all.deb ...
Moving old data out of the way
Unpacking ucf (3.0043) ...
Selecting previously unselected package libpng16-16:amd64.
Preparing to unpack .../04-libpng16-16_1.6.37-3ubuntu0.6_amd64.deb ...
Unpacking libpng16-16:amd64 (1.6.37-3ubuntu0.6) ...
Selecting previously unselected package libxau6:amd64.
Preparing to unpack .../05-libxau6_1%3a1.0.9-1build5_amd64.deb ...
Unpacking libxau6:amd64 (1:1.0.9-1build5) ...
Selecting previously unselected package libxdmcp6:amd64.
Preparing to unpack .../06-libxdmcp6_1%3a1.1.3-0ubuntu5_amd64.deb ...
Unpacking libxdmcp6:amd64 (1:1.1.3-0ubuntu5) ...
Selecting previously unselected package libxcb1:amd64.
Preparing to unpack .../07-libxcb1_1.14-3ubuntu3_amd64.deb ...
Unpacking libxcb1:amd64 (1.14-3ubuntu3) ...
Selecting previously unselected package libx11-data.
Preparing to unpack .../08-libx11-data_2%3a1.7.5-1ubuntu0.3_all.deb ...
Unpacking libx11-data (2:1.7.5-1ubuntu0.3) ...
Selecting previously unselected package libx11-6:amd64.
Preparing to unpack .../09-libx11-6_2%3a1.7.5-1ubuntu0.3_amd64.deb ...
Unpacking libx11-6:amd64 (2:1.7.5-1ubuntu0.3) ...
Selecting previously unselected package manpages.
Preparing to unpack .../10-manpages_5.10-1ubuntu1_all.deb ...
Unpacking manpages (5.10-1ubuntu1) ...
Selecting previously unselected package binutils-common:amd64.
Preparing to unpack .../11-binutils-common_2.38-4ubuntu2.12_amd64.deb ...
Unpacking binutils-common:amd64 (2.38-4ubuntu2.12) ...
Selecting previously unselected package libbinutils:amd64.
Preparing to unpack .../12-libbinutils_2.38-4ubuntu2.12_amd64.deb ...
Unpacking libbinutils:amd64 (2.38-4ubuntu2.12) ...
Selecting previously unselected package libctf-nobfd0:amd64.
Preparing to unpack .../13-libctf-nobfd0_2.38-4ubuntu2.12_amd64.deb ...
Unpacking libctf-nobfd0:amd64 (2.38-4ubuntu2.12) ...
Selecting previously unselected package libctf0:amd64.
Preparing to unpack .../14-libctf0_2.38-4ubuntu2.12_amd64.deb ...
Unpacking libctf0:amd64 (2.38-4ubuntu2.12) ...
Selecting previously unselected package binutils-x86-64-linux-gnu.
Preparing to unpack .../15-binutils-x86-64-linux-gnu_2.38-4ubuntu2.12_amd64.deb ...
Unpacking binutils-x86-64-linux-gnu (2.38-4ubuntu2.12) ...
Selecting previously unselected package binutils.
Preparing to unpack .../16-binutils_2.38-4ubuntu2.12_amd64.deb ...
Unpacking binutils (2.38-4ubuntu2.12) ...
Selecting previously unselected package gcc-11-base:amd64.
Preparing to unpack .../17-gcc-11-base_11.4.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking gcc-11-base:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libisl23:amd64.
Preparing to unpack .../18-libisl23_0.24-2build1_amd64.deb ...
Unpacking libisl23:amd64 (0.24-2build1) ...
Selecting previously unselected package libmpfr6:amd64.
Preparing to unpack .../19-libmpfr6_4.1.0-3build3_amd64.deb ...
Unpacking libmpfr6:amd64 (4.1.0-3build3) ...
Selecting previously unselected package libmpc3:amd64.
Preparing to unpack .../20-libmpc3_1.2.1-2build1_amd64.deb ...
Unpacking libmpc3:amd64 (1.2.1-2build1) ...
Selecting previously unselected package cpp-11.
Preparing to unpack .../21-cpp-11_11.4.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking cpp-11 (11.4.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package cpp.
Preparing to unpack .../22-cpp_4%3a11.2.0-1ubuntu1_amd64.deb ...
Unpacking cpp (4:11.2.0-1ubuntu1) ...
Selecting previously unselected package fonts-dejavu-core.
Preparing to unpack .../23-fonts-dejavu-core_2.37-2build1_all.deb ...
Unpacking fonts-dejavu-core (2.37-2build1) ...
Selecting previously unselected package fontconfig-config.
Preparing to unpack .../24-fontconfig-config_2.13.1-4.2ubuntu5_all.deb ...
Unpacking fontconfig-config (2.13.1-4.2ubuntu5) ...
Selecting previously unselected package libcc1-0:amd64.
Preparing to unpack .../25-libcc1-0_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libcc1-0:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libgomp1:amd64.
Preparing to unpack .../26-libgomp1_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libgomp1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libitm1:amd64.
Preparing to unpack .../27-libitm1_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libitm1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libatomic1:amd64.
Preparing to unpack .../28-libatomic1_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libatomic1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libasan6:amd64.
Preparing to unpack .../29-libasan6_11.4.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libasan6:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package liblsan0:amd64.
Preparing to unpack .../30-liblsan0_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking liblsan0:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libtsan0:amd64.
Preparing to unpack .../31-libtsan0_11.4.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libtsan0:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libubsan1:amd64.
Preparing to unpack .../32-libubsan1_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libubsan1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libquadmath0:amd64.
Preparing to unpack .../33-libquadmath0_12.3.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libquadmath0:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package libgcc-11-dev:amd64.
Preparing to unpack .../34-libgcc-11-dev_11.4.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking libgcc-11-dev:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package gcc-11.
Preparing to unpack .../35-gcc-11_11.4.0-1ubuntu1~22.04.3_amd64.deb ...
Unpacking gcc-11 (11.4.0-1ubuntu1~22.04.3) ...
Selecting previously unselected package gcc.
Preparing to unpack .../36-gcc_4%3a11.2.0-1ubuntu1_amd64.deb ...
Unpacking gcc (4:11.2.0-1ubuntu1) ...
Selecting previously unselected package libbrotli1:amd64.
Preparing to unpack .../37-libbrotli1_1.0.9-2build6_amd64.deb ...
Unpacking libbrotli1:amd64 (1.0.9-2build6) ...
Selecting previously unselected package libc-dev-bin.
Preparing to unpack .../38-libc-dev-bin_2.35-0ubuntu3.15_amd64.deb ...
Unpacking libc-dev-bin (2.35-0ubuntu3.15) ...
Selecting previously unselected package libfreetype6:amd64.
Preparing to unpack .../39-libfreetype6_2.11.1+dfsg-1ubuntu0.3_amd64.deb ...
Unpacking libfreetype6:amd64 (2.11.1+dfsg-1ubuntu0.3) ...
Selecting previously unselected package libfontconfig1:amd64.
Preparing to unpack .../40-libfontconfig1_2.13.1-4.2ubuntu5_amd64.deb ...
Unpacking libfontconfig1:amd64 (2.13.1-4.2ubuntu5) ...
Selecting previously unselected package libjpeg-turbo8:amd64.
Preparing to unpack .../41-libjpeg-turbo8_2.1.2-0ubuntu1_amd64.deb ...
Unpacking libjpeg-turbo8:amd64 (2.1.2-0ubuntu1) ...
Selecting previously unselected package libjpeg8:amd64.
Preparing to unpack .../42-libjpeg8_8c-2ubuntu10_amd64.deb ...
Unpacking libjpeg8:amd64 (8c-2ubuntu10) ...
Selecting previously unselected package libdeflate0:amd64.
Preparing to unpack .../43-libdeflate0_1.10-2_amd64.deb ...
Unpacking libdeflate0:amd64 (1.10-2) ...
Selecting previously unselected package libjbig0:amd64.
Preparing to unpack .../44-libjbig0_2.1-3.1ubuntu0.22.04.1_amd64.deb ...
Unpacking libjbig0:amd64 (2.1-3.1ubuntu0.22.04.1) ...
Selecting previously unselected package libwebp7:amd64.
Preparing to unpack .../45-libwebp7_1.2.2-2ubuntu0.22.04.2_amd64.deb ...
Unpacking libwebp7:amd64 (1.2.2-2ubuntu0.22.04.2) ...
Selecting previously unselected package libtiff5:amd64.
Preparing to unpack .../46-libtiff5_4.3.0-6ubuntu0.13_amd64.deb ...
Unpacking libtiff5:amd64 (4.3.0-6ubuntu0.13) ...
Selecting previously unselected package libxpm4:amd64.
Preparing to unpack .../47-libxpm4_1%3a3.5.12-1ubuntu0.22.04.3_amd64.deb ...
Unpacking libxpm4:amd64 (1:3.5.12-1ubuntu0.22.04.3) ...
Selecting previously unselected package libgd3:amd64.
Preparing to unpack .../48-libgd3_2.3.0-2ubuntu2.3_amd64.deb ...
Unpacking libgd3:amd64 (2.3.0-2ubuntu2.3) ...
Selecting previously unselected package libc-devtools.
Preparing to unpack .../49-libc-devtools_2.35-0ubuntu3.15_amd64.deb ...
Unpacking libc-devtools (2.35-0ubuntu3.15) ...
Selecting previously unselected package linux-libc-dev:amd64.
Preparing to unpack .../50-linux-libc-dev_5.15.0-194.204_amd64.deb ...
Unpacking linux-libc-dev:amd64 (5.15.0-194.204) ...
Selecting previously unselected package libcrypt-dev:amd64.
Preparing to unpack .../51-libcrypt-dev_1%3a4.4.27-1_amd64.deb ...
Unpacking libcrypt-dev:amd64 (1:4.4.27-1) ...
Selecting previously unselected package rpcsvc-proto.
Preparing to unpack .../52-rpcsvc-proto_1.4.2-0ubuntu6_amd64.deb ...
Unpacking rpcsvc-proto (1.4.2-0ubuntu6) ...
Selecting previously unselected package libtirpc-dev:amd64.
Preparing to unpack .../53-libtirpc-dev_1.3.2-2ubuntu0.1_amd64.deb ...
Unpacking libtirpc-dev:amd64 (1.3.2-2ubuntu0.1) ...
Selecting previously unselected package libnsl-dev:amd64.
Preparing to unpack .../54-libnsl-dev_1.3.0-2build2_amd64.deb ...
Unpacking libnsl-dev:amd64 (1.3.0-2build2) ...
Selecting previously unselected package libc6-dev:amd64.
Preparing to unpack .../55-libc6-dev_2.35-0ubuntu3.15_amd64.deb ...
Unpacking libc6-dev:amd64 (2.35-0ubuntu3.15) ...
Selecting previously unselected package manpages-dev.
Preparing to unpack .../56-manpages-dev_5.10-1ubuntu1_all.deb ...
Unpacking manpages-dev (5.10-1ubuntu1) ...
Setting up libexpat1:amd64 (2.4.7-1ubuntu0.8) ...
Setting up gcc-11-base:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Setting up libxau6:amd64 (1:1.0.9-1build5) ...
Setting up manpages (5.10-1ubuntu1) ...
Setting up libbrotli1:amd64 (1.0.9-2build6) ...
Setting up binutils-common:amd64 (2.38-4ubuntu2.12) ...
Setting up libdeflate0:amd64 (1.10-2) ...
Setting up linux-libc-dev:amd64 (5.15.0-194.204) ...
Setting up libctf-nobfd0:amd64 (2.38-4ubuntu2.12) ...
Setting up libgomp1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Setting up libjbig0:amd64 (2.1-3.1ubuntu0.22.04.1) ...
Setting up libasan6:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Setting up libtirpc-dev:amd64 (1.3.2-2ubuntu0.1) ...
Setting up rpcsvc-proto (1.4.2-0ubuntu6) ...
Setting up libx11-data (2:1.7.5-1ubuntu0.3) ...
Setting up libmpfr6:amd64 (4.1.0-3build3) ...
Setting up libquadmath0:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Setting up libpng16-16:amd64 (1.6.37-3ubuntu0.6) ...
Setting up libmpc3:amd64 (1.2.1-2build1) ...
Setting up libatomic1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Setting up fonts-dejavu-core (2.37-2build1) ...
Setting up ucf (3.0043) ...
debconf: unable to initialize frontend: Dialog
debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78.)
debconf: falling back to frontend: Readline
debconf: unable to initialize frontend: Readline
debconf: (Can't locate Term/ReadLine.pm in @INC (you may need to install the Term::ReadLine module) (@INC contains: /etc/perl /usr/local/lib/x86_64-linux-gnu/perl/5.34.0 /usr/local/share/perl/5.34.0 /usr/lib/x86_64-linux-gnu/perl5/5.34 /usr/share/perl5 /usr/lib/x86_64-linux-gnu/perl-base /usr/lib/x86_64-linux-gnu/perl/5.34 /usr/share/perl/5.34 /usr/local/lib/site_perl) at /usr/share/perl5/Debconf/FrontEnd/Readline.pm line 7.)
debconf: falling back to frontend: Teletype
Setting up libjpeg-turbo8:amd64 (2.1.2-0ubuntu1) ...
Setting up libwebp7:amd64 (1.2.2-2ubuntu0.22.04.2) ...
Setting up libubsan1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Setting up libmd0:amd64 (1.0.4-1build1) ...
Setting up libnsl-dev:amd64 (1.3.0-2build2) ...
Setting up libcrypt-dev:amd64 (1:4.4.27-1) ...
Setting up libbinutils:amd64 (2.38-4ubuntu2.12) ...
Setting up libisl23:amd64 (0.24-2build1) ...
Setting up libc-dev-bin (2.35-0ubuntu3.15) ...
Setting up libbsd0:amd64 (0.11.5-1) ...
Setting up libcc1-0:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Setting up liblsan0:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Setting up libitm1:amd64 (12.3.0-1ubuntu1~22.04.3) ...
Setting up libtsan0:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Setting up libctf0:amd64 (2.38-4ubuntu2.12) ...
Setting up libjpeg8:amd64 (8c-2ubuntu10) ...
Setting up cpp-11 (11.4.0-1ubuntu1~22.04.3) ...
Setting up manpages-dev (5.10-1ubuntu1) ...
Setting up libxdmcp6:amd64 (1:1.1.3-0ubuntu5) ...
Setting up libxcb1:amd64 (1.14-3ubuntu3) ...
Setting up fontconfig-config (2.13.1-4.2ubuntu5) ...
Setting up libfreetype6:amd64 (2.11.1+dfsg-1ubuntu0.3) ...
Setting up libgcc-11-dev:amd64 (11.4.0-1ubuntu1~22.04.3) ...
Setting up cpp (4:11.2.0-1ubuntu1) ...
Setting up libc6-dev:amd64 (2.35-0ubuntu3.15) ...
Setting up libx11-6:amd64 (2:1.7.5-1ubuntu0.3) ...
Setting up libtiff5:amd64 (4.3.0-6ubuntu0.13) ...
Setting up libfontconfig1:amd64 (2.13.1-4.2ubuntu5) ...
Setting up binutils-x86-64-linux-gnu (2.38-4ubuntu2.12) ...
Setting up libxpm4:amd64 (1:3.5.12-1ubuntu0.22.04.3) ...
Setting up binutils (2.38-4ubuntu2.12) ...
Setting up libgd3:amd64 (2.3.0-2ubuntu2.3) ...
Setting up gcc-11 (11.4.0-1ubuntu1~22.04.3) ...
Setting up libc-devtools (2.35-0ubuntu3.15) ...
Setting up gcc (4:11.2.0-1ubuntu1) ...
Processing triggers for libc-bin (2.35-0ubuntu3.8) ...
Saving image...
Image saved, took 1.12s

Built image im-5brGOZd92uODnlCr0MFCus in 30.51s


Building image im-iIGMjl3LCzMHpUTcmOzqQv

=> Step 0: FROM base

=> Step 1: ENV HF_HUB_OFFLINE=1
Saving image...
Image saved, took 619.97ms

Built image im-iIGMjl3LCzMHpUTcmOzqQv in 2.56s


✓ Created objects.
├── 🔨 Created mount 
│   /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab/modal_app/r4_bench.py
├── 🔨 Created mount 
│   /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab/src/s2slab
├── 🔨 Created function download_weights.
├── 🔨 Created mount 
│   /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab/probes/endpointing
├── 🔨 Created mount /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab/evalset
├── 🔨 Created function bench.
└── 🔨 Created function evaluate.

==========
== CUDA ==
==========

CUDA Version 12.6.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.


[1ALoading weights:   0%|          | 0/876 [00:00<?, ?it/s]
[1ALoading weights:   1%|          | 5/876 [00:00<00:17, 49.34it/s]
[1ALoading weights:   3%|▎         | 26/876 [00:00<00:07, 110.02it/s]
[1ALoading weights:  23%|██▎       | 205/876 [00:00<00:00, 686.09it/s]
[1ALoading weights:  31%|███▏      | 274/876 [00:00<00:01, 343.08it/s]
[1ALoading weights:  37%|███▋      | 323/876 [00:01<00:03, 161.29it/s]
[1ALoading weights:  41%|████      | 357/876 [00:02<00:03, 132.33it/s]
[1ALoading weights:  44%|████▎     | 382/876 [00:02<00:04, 112.17it/s]
[1ALoading weights:  46%|████▌     | 401/876 [00:02<00:04, 111.24it/s]
[1ALoading weights:  48%|████▊     | 418/876 [00:02<00:04, 108.22it/s]
[1ALoading weights:  49%|████▉     | 433/876 [00:02<00:04, 107.67it/s]
[1ALoading weights:  51%|█████     | 446/876 [00:03<00:04, 100.98it/s]
[1ALoading weights:  52%|█████▏    | 458/876 [00:03<00:04, 92.57it/s] 
[1ALoading weights:  54%|█████▎    | 470/876 [00:03<00:04, 97.13it/s]
[1ALoading weights:  55%|█████▌    | 484/876 [00:03<00:03, 103.28it/s]
[1ALoading weights:  57%|█████▋    | 496/876 [00:03<00:03, 101.83it/s]
[1ALoading weights:  58%|█████▊    | 509/876 [00:03<00:03, 108.00it/s]
[1ALoading weights:  59%|█████▉    | 521/876 [00:03<00:03, 99.61it/s] 
[1ALoading weights:  61%|██████    | 532/876 [00:05<00:15, 22.48it/s]
[1ALoading weights:  62%|██████▏   | 540/876 [00:05<00:12, 25.85it/s]
[1ALoading weights:  63%|██████▎   | 550/876 [00:05<00:10, 32.14it/s]
[1ALoading weights:  64%|██████▎   | 558/876 [00:05<00:08, 36.91it/s]
[1ALoading weights:  65%|██████▍   | 567/876 [00:05<00:07, 43.88it/s]
[1ALoading weights:  66%|██████▌   | 577/876 [00:05<00:05, 52.69it/s]
[1ALoading weights:  67%|██████▋   | 591/876 [00:06<00:04, 68.82it/s]
[1ALoading weights:  69%|██████▉   | 603/876 [00:06<00:03, 76.79it/s]
[1ALoading weights:  70%|███████   | 614/876 [00:06<00:03, 81.78it/s]
[1ALoading weights:  92%|█████████▏| 805/876 [00:06<00:00, 516.12it/s]
[1ALoading weights: 100%|██████████| 876/876 [00:06<00:00, 137.01it/s]
/usr/local/lib/python3.12/site-packages/torch/nn/modules/rnn.py:1011: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  super().__init__("LSTM", *args, **kwargs)
/usr/local/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:145: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
/usr/local/lib/python3.12/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
  warnings.warn(
[transformers] Model config: tts_text_start_token_id must be `None` or an integer within the vocabulary (between 0 and 8447), got 151860. This may result in unexpected behavior.
[transformers] Model config: tts_text_end_token_id must be `None` or an integer within the vocabulary (between 0 and 8447), got 151861. This may result in unexpected behavior.
[transformers] Model config: tts_text_pad_token_id must be `None` or an integer within the vocabulary (between 0 and 8447), got 151859. This may result in unexpected behavior.
[transformers] Model config: vision_start_token_id must be `None` or an integer within the vocabulary (between 0 and 8447), got 151652. This may result in unexpected behavior.
[transformers] Model config: vision_end_token_id must be `None` or an integer within the vocabulary (between 0 and 8447), got 151653. This may result in unexpected behavior.
[transformers] Model config: audio_start_token_id must be `None` or an integer within the vocabulary (between 0 and 8447), got 151647. This may result in unexpected behavior.
[transformers] Model config: audio_end_token_id must be `None` or an integer within the vocabulary (between 0 and 8447), got 151648. This may result in unexpected behavior.

[1ALoading weights:   0%|          | 0/2447 [00:00<?, ?it/s]
[1ALoading weights:   1%|          | 20/2447 [00:00<00:13, 182.58it/s]
[1ALoading weights:   2%|▏         | 41/2447 [00:00<00:13, 180.39it/s]
[1ALoading weights:   4%|▎         | 87/2447 [00:00<00:07, 298.59it/s]
[1ALoading weights:   6%|▌         | 135/2447 [00:00<00:06, 361.98it/s]
[1ALoading weights:   7%|▋         | 182/2447 [00:00<00:05, 391.19it/s]
[1ALoading weights:   9%|▉         | 232/2447 [00:00<00:05, 416.70it/s]
[1ALoading weights:  12%|█▏        | 286/2447 [00:00<00:04, 453.45it/s]
[1ALoading weights:  15%|█▍        | 358/2447 [00:00<00:04, 499.22it/s]
[1ALoading weights:  17%|█▋        | 408/2447 [00:01<00:07, 265.44it/s]
[1ALoading weights:  18%|█▊        | 447/2447 [00:01<00:09, 210.55it/s]
[1ALoading weights:  20%|█▉        | 478/2447 [00:01<00:09, 196.92it/s]
[1ALoading weights:  21%|██        | 504/2447 [00:01<00:11, 172.63it/s]
[1ALoading weights:  21%|██▏       | 526/2447 [00:02<00:11, 173.97it/s]
[1ALoading weights:  22%|██▏       | 547/2447 [00:02<00:11, 161.93it/s]
[1ALoading weights:  23%|██▎       | 566/2447 [00:02<00:12, 151.84it/s]
[1ALoading weights:  24%|██▍       | 583/2447 [00:02<00:12, 149.83it/s]
[1ALoading weights:  26%|██▌       | 628/2447 [00:02<00:08, 212.71it/s]
[1ALoading weights:  28%|██▊       | 685/2447 [00:02<00:06, 291.23it/s]
[1ALoading weights:  30%|███       | 742/2447 [00:02<00:04, 354.50it/s]
[1ALoading weights:  33%|███▎      | 797/2447 [00:02<00:04, 398.71it/s]
[1ALoading weights:  35%|███▍      | 846/2447 [00:03<00:03, 419.84it/s]
[1ALoading weights:  37%|███▋      | 911/2447 [00:03<00:03, 478.23it/s]
[1ALoading weights:  39%|███▉      | 961/2447 [00:03<00:03, 425.76it/s]
[1ALoading weights:  44%|████▍     | 1086/2447 [00:03<00:02, 639.61it/s]
[1ALoading weights:  50%|█████     | 1229/2447 [00:03<00:01, 853.28it/s]
[1ALoading weights:  54%|█████▍    | 1321/2447 [00:03<00:02, 538.52it/s]
[1ALoading weights:  57%|█████▋    | 1394/2447 [00:04<00:02, 450.84it/s]
[1ALoading weights: 100%|██████████| 2447/2447 [00:04<00:00, 589.26it/s]
[transformers] [1mQwen2_5OmniForConditionalGeneration LOAD REPORT[0m from: Qwen/Qwen2.5-Omni-7B
Key                                                | Status     |  | 
---------------------------------------------------+------------+--+-
token2wav.code2wav_dit_model.rotary_embed.inv_freq | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
r3-gpu: R3 speech-LLM: Qwen2-Audio-7B-Instruct hears the turn -> Kokoro-82M; no transcript (bf16, CUDA)
commit 339092a · Modal NVIDIA L40S, 580.95.05, one container: r3-gpu then r4-gpu · run 2026-09-23 · eval set built 2026-09-15, seed 20260915
load 34.1 s, once, before the first turn · peak GPU memory 16,986 MB

### Processing time

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 699 | — | — |
| accented | 50 | 688 | — | — |
| disfluent | 50 | 750 | — | — |
| slow | 50 | 629 | — | — |
| all | 200 | 688 | 1286 | — |

### First audio: when the first piece of the reply existed

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 699 | — | — |
| accented | 50 | 688 | — | — |
| disfluent | 50 | 750 | — | — |
| slow | 50 | 629 | — | — |
| all | 200 | 688 | 1286 | — |

### First sound: first audio plus the silence before the first word

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 1009 | — | — |
| accented | 50 | 991 | — | — |
| disfluent | 50 | 1056 | — | — |
| slow | 50 | 929 | — | — |
| all | 200 | 991 | 1603 | — |

### Where the time went

| stage | p50 ms | p90 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| first sound | 991.2 | 1603.1 | — | 200 |
| total | 687.9 | 1285.7 | — | 200 |
| first audio | 687.9 | 1285.7 | — | 200 |
| llm | 619.7 | 1203.5 | — | 200 |
| tts | 70.4 | 88.0 | — | 200 |
| (untimed) | 0.0 | 0.0 | — | 200 |

### Word error rate

no transcript in the loop: the audio goes straight into the LLM, so there is nothing to score

### asked to transcribe, outside the loop (untimed; not what the reply was made from)

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| accented | 50 | 884 | 25.2% | 23.3% |
| clean | 50 | 846 | 1.9% | 1.9% |
| disfluent | 50 | 1039 | 9.9% | 8.7% |
| slow | 50 | 846 | 1.8% | 1.8% |
| **all** | 200 | 3615 | 9.9% | 9.1% |

reply leading silence: p50   300.0  p90   320.0  p99   340.0  [n=200, unsupported: p99]
playback stall: p50     0.0  p90     0.0  p99     0.0  [n=200, unsupported: p99]  (0 of 200 turns ran dry)

saved -> runs/r3-gpu/evalset-modal-l40s.json

r4-gpu: R4 S2S half-duplex: Qwen2.5-Omni-7B hears the turn and speaks the reply itself; no transcript, no TTS (bf16, CUDA)
commit 339092a · Modal NVIDIA L40S, 580.95.05, one container: r3-gpu then r4-gpu · run 2026-09-23 · eval set built 2026-09-15, seed 20260915
load 25.1 s, once, before the first turn · peak GPU memory 22,518 MB

### Processing time

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 9037 | — | — |
| accented | 50 | 8766 | — | — |
| disfluent | 50 | 9750 | — | — |
| slow | 50 | 9085 | — | — |
| all | 200 | 9166 | 17325 | — |

### First audio: when the first piece of the reply existed

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 9037 | — | — |
| accented | 50 | 8766 | — | — |
| disfluent | 50 | 9750 | — | — |
| slow | 50 | 9085 | — | — |
| all | 200 | 9166 | 17325 | — |

### First sound: first audio plus the silence before the first word

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 9237 | — | — |
| accented | 50 | 8986 | — | — |
| disfluent | 50 | 9947 | — | — |
| slow | 50 | 9285 | — | — |
| all | 200 | 9366 | 17523 | — |

### Where the time went

| stage | p50 ms | p90 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| first sound | 9365.8 | 17522.6 | — | 200 |
| s2s | 9165.8 | 17324.5 | — | 200 |
| total | 9165.8 | 17324.6 | — | 200 |
| first audio | 9165.8 | 17324.6 | — | 200 |
| (untimed) | 0.1 | 0.1 | — | 200 |

### Word error rate

no transcript in the loop: the audio goes straight into the LLM, so there is nothing to score

reply leading silence: p50   200.0  p90   220.0  p99   220.0  [n=200, unsupported: p99]
playback stall: p50     0.0  p90     0.0  p99     0.0  [n=200, unsupported: p99]  (0 of 200 turns ran dry)

saved -> runs/r4-gpu/evalset-modal-l40s.json

torch 2.14.0+cu126, torchvision 0.29.0+cu126, transformers 5.17.0
2425 s inside the function, 2433 s end to end (queue, cold start and run): about $1.318 at the L40S rate. Modal's dashboard has the exact bill.
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/ramcharanmc/main/ap-SmPdw14AmM4a7XR0BiNNnd

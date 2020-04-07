# COVID-2019 Prediction With Federated Learning



## 1. Introduction

We provide a simple but efficient Federated Learning(FL) framework for researchers to  train the model on several remote machines.

Like most C/S structures, the framework consists of two parts: Server side, and Clients side. Take hospitals for example, if you want to train a federated model, you could deploy the client part to the hospitals and run the server part on your center machine.

Once the code is run, the hospitals will train their own model locally and  transmit the parameters to  the server and server will aggregate all these parameters from several clients. Then the server will distribute the newly aggregated parameters to each clients joined the FL process. Iterate this process for some pre-set rounds or the aggregated model reached the desired accuracy.



### 1.1 Server side

#### 1.1.1 Scripts

1. `server_register.py`: listen to the port, if a client connect, verify the client. If the clients pass the verification by its user name and password, then the server will add this client's IP to the qualified IP list.   

2. `server_send.py`: listen to the port, verify the client IP by qulified IP list, if a verified client connect, send the aggregated parameters to the client.

3. `server_recv.py`: listen to the port, if  a verified client connect, receive the updated parameters from the client.

4. `aggregate.py`: functional file, provides a aggregation function.

5. `server_main.py`: main script of server side.

   <img src="./pic/server_main.jpg" style="zoom: 50%;" />



#### 1.1.2 Configurations

1. `server_config.json`: configuration file of server, contain some key parameters for your script running. There are some examples below.

   >{
   >
   >​	"server_ip": "0.0.0.0", 
   >
   >​	"send_server_port": 8091, 
   >
   >​	"recv_server_port": 8090, 
   >
   >​	"register_port": 2333, 
   >
   >​	 "min_clients": 1, 
   >
   >​	"max_iterations": 10,  
   >
   >​	"model": "./server_data/model.py", 
   >
   >​	"weight": "./server_data/30_epoch.pth", 
   >
   >​	"weight_directory": "./client_data/"
   >
   >}

2. `server_users.json`: a  `username:password` dictionary kept by server, `server_register.py` use it to verify whether a client is qualified.

   >{
   >    "Bob": "123456",
   >    "Alan": "123456",
   >    "John": "123456"
   >}



### 1.2 Client Side

#### 1.2.1 Scripts

1. `client_register.py`: register current IP to the server.
2. `client_recv.py`: receive the initial/aggregated model(parameterss) from the server.
3. `client_send.py`: send the updated model to the server.



#### 1.2.2 Configuration

1. `config_client.json`: configuration file o the client.

>{
>    "username": "Alan",
>    "password": "123456",
>    "server_ip": "10.0.0.5",
>    "send_server_port": 8090,
>    "recv_server_port": 8091,
>    "register_server_port": 2333,
>    "buffsize": 1024,
>    "model_path": "/home/user/fed_learing/model.py",
>    "weight_pth": "/home/user/fed_learing/weight.pth"
>}



## 2. Installation

You could run this command `git clone https://github.com/HUST-EIC-AI-LAB/COVID-19-Fedrated-Learinig.git`  to deploy your own FL task. But there are some dependencies should be pre-installed, like pytorch and CUDA if you'd like to train on GPU.



#### Pre Installation

Run `pip install -r requirement.txt` to install the needed dependencies.

**Notice:** </br>
In `requirement.txt`, we use torch matches `cuda == 9.2.` 
If there is problem about torch when you try to run, it may caused by Version mismatch between torch and cuda. please check your cuda version with  `cat /usr/local/cuda/version.txt`.

 And download the correct version of Pytorch from the official website([https://pytorch.org/](https://pytorch.org/ "Pytorch")).



## 3. What should user do

### 3.1 Client: On client machine

1. Modify the parameters on the configuration file `./config/config_client.json`  </br>
   Args:</br>
   &emsp;&emsp;1). `username`: Account information distributed by the server.</br>
   &emsp;&emsp;2). `password`: Account information distributed by the server.</br> 
   &emsp;&emsp;3). `server_ip`: Server IP which achieve FL process management and model parameter aggregation.</br>
   &emsp;&emsp;4). `send_server_port`: Send updated model parameter difference from the port.</br> 
   &emsp;&emsp;5). `recv_server_port`: Receive file of model structure and initial parameters from the port.</br> 
   &emsp;&emsp;6). `register_server_port`: Registry the client IP on server before starting training.</br>

2. run `python download.py`</br>
   **TODO:**
   Download model structure and initial weight info from server, the two files will be save on `./download`.

3. Modify the parameters on the configuration file `./config/config.json`</br>
   Args:</br>
      	&emsp;&emsp;1). `load_model`: the path of model structure file.</br>
      	&emsp;&emsp; eg. `"./download/weight_v1.pth"`</br>
      	&emsp;&emsp;2). `train_data_dir`: the path of your own data file(The home directory where the CT image is located).</br>
      	&emsp;&emsp; eg. `"./mnt/data/dataset"`</br>
      	&emsp;&emsp;3). `train_df_csv`: the path of csv CSV list of training sets.</br>
      	&emsp;&emsp; eg. `"./utils/train_clean_data.csv"`</br>
      	&emsp;&emsp;4). `labels_train_df_csv`: the path of CSV list of training sets.</br>
      	&emsp;&emsp; eg. `"./utils/train_clean_data.csv"`</br>
      	&emsp;&emsp;5). `use_cuda`: use cuda to train or not.</br>
      	&emsp;&emsp;6). `epoch`: epoch of local training. (We recommend using the same training epoch on subdevices, eg. epoch=5)</br>
      	&emsp;&emsp;7). `num_workers`: dataloader will creat `num_workers` subThread to load data. </br>
      	&emsp;&emsp;8). `lr`: We recommed using the small learning rate in the training, eg 1e-3. With SGD optimizer.</br>
      	&emsp;&emsp;9). `momentum`: default: 0.9.

4. run `python train.py`</br>
   **TODO:**</br>
   Start training on you local device with local data. Then, the updated model will be saved to `./checkpoint` after each epoch training finished. When the whole training process finished, the process will send the difference between initial model and updated model to server. For other clients, they have no way to get the local data on this client device, and even they can not understand the meaning of Parameter difference, because these parameters are not really meaningful in a way. So that, client privacy will be protected.</br>
   Notice: If there is a connection problem on the upload, you could finish it yourself.
   		look at the name of model difference file in `./checkpoint`, we call its path as file_path.
   		try to execute `python upload.py file_path` to finish uploading.



### 3.2 Server: On server machine

1. run `python server_main.py`



## 4. How to upload and download parameters

We know that two processes if you need one of the most basic premise of communication can only sign a process, we can use the PID in the local process communication to only marked a process, but only in the PID, the only local, two processes of network PID conflict odds is very big, diameter at this time we need to find something else to do it. 

We know the IP address of the IP layer can only mark the host, and the TCP layer protocol and port number identifying the host can be the only one process, so that we can use the IP address + port number + agreement only a process identified in the network. Once the processes in the network can be uniquely identified, they can communicate using sockets. 

We've designed two ways to upload the parameters to the server:

1. Full Automatically: If all the process run successfully and don't have any errors, the parameter will be uploaded to the server after the training process finished.
2. Self-automatically: Otherwise, client could upload the parameters manually, and check if there is newly aggregated parameters distributed by the server.

​	






# COVID-2019 Prediction With Federated Learning



## 1. Introduction

We provide a simple but efficient Federated Learning(FL) framework for researchers to train the model on remote machines.

Like most C/S structures, the framework consists of two parts: Server side, and Client side. To apply this framwork in real scenarios, take hospitals for example, the client part could be deployed on the machines of the hospital end (Client) where the federated model is trained locally, while the server part is established on the center machine (Server).

Once the scripts are executed, the hospitals will train their own models locally and transmit the parameters to the server where the server aggregates all parameters collected from the clients. Then the server will distribute the newly aggregated parameters to each clients joined the FL process. This process will iterate for some pre-set rounds before accuracy of the aggregated model reached the desired level.



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
   >​ "server_ip": "0.0.0.0", 
   >
   >​ "send_server_port": 8091, 
   >
   >​ "recv_server_port": 8090, 
   >
   >​ "register_port": 2333, 
   >
   >​  "min_clients": 1, 
   >
   >​ "max_iterations": 10,  
   >
   >​ "model": "./server_data/model.py", 
   >
   >​ "weight": "./server_data/30_epoch.pth", 
   >
   >​ "weight_directory": "./client_data/"
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
</br>
</br>
</br>

## 2. FL framework Installation

### Install FL framework from Github 
You could run this command `git clone https://github.com/HUST-EIC-AI-LAB/COVID-19-Fedrated-Learinig.git`  to deploy your own FL task. 

#### Dependencies installation
Some dependencies may need to be pre-installed, e.g. pytorch and CUDA, before you can train on GPU.
Run `pip install -r requirement.txt` to install the needed dependencies.

**Notice:** </br>
In `requirement.txt`, we use torch matches `cuda == 9.2.` 

If there are problems while using torch, it may be caused by version mismatch between torch and CUDA, please check your CUDA version by `cat /usr/local/cuda/version.txt`, and download the correct version of Pytorch from the official website([https://pytorch.org/](https://pytorch.org/ "Pytorch")).
</br>
</br>
</br>
## 3. Implementation of FL

   ### 3.1 Client: on client machines

   #### 1. Client Registration</br>
   Modify the parameters on the configuration file `./config/config_client.json`  </br>
   Args:</br>
   &emsp;&emsp;1). `username`: Account information distributed by the server.</br>
   &emsp;&emsp;2). `password`: Account information distributed by the server.</br> 
   &emsp;&emsp;3). `server_ip`: Server IP which achieve FL process management and model parameter aggregation.</br>
   &emsp;&emsp;4). `send_server_port`: Send updated model parameter difference from the port.</br> 
   &emsp;&emsp;5). `recv_server_port`: Receive file of model structure and initial parameters from the port.</br> 
   &emsp;&emsp;6). `register_server_port`: Registry the client IP on server before starting training.</br>

   
   
   #### 2. Model Download & Configuration </br>
   Modify the parameters on the configuration file `./config/config.json`</br>
   Args:</br>
         &emsp;&emsp;1). `load_model`: the path of model structure file.</br>
         &emsp;&emsp; eg. `"./download/weight_v1.pth"`</br>
         &emsp;&emsp;2). `train_data_dir`: the path of your own data file(The home directory where the CT image is located).     </br>
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

   

   #### 3. Train Locally </br> 
   run `python train.py`</br>


   #### 4. Upload Parameters
   To make things easier, we've integrated all processes ont the client side, including training and uploading, into one single file(`python train.py`):

   ​  After the process of uploading a local training is completed, the client will constantly ask the sever for the newest merged model. If the server completes the merge operation, the new model will be sent to the client. After receiving the new model, the client starts the next round of local training ; If the sever has not completed the merg and the client did not get anything from the server, it will send a request again after a certain time.

   ​  Start training on you local device with local data. The updated model will be saved to `./checkpoint` after each epoch training is finished. When the whole training process is finished, the process will send the difference between the initial model and the updated model to the server. For other clients, there is no way for them to get the local data on this client device, and even they can not understand the meaning of Parameter difference, because these parameters are not really meaningful in a way. So that, client privacy will be protected.</br>

   **Notice:**</br>
   If there is a connection problem on the upload, you could finish it yourself.
   Find the model difference file in `./checkpoint`(`file_path` for example).
   Execute `python upload.py file_path` to finish uploading.



### 3.2 Server: on server machines

#### 1. Execute server program
Run `python server_main.py`
</br>
</br>
</br>

## 4. More about downloading & uploading parameters

The parameters uploaded to the central server in the process are actually the "gradient difference",  **not the image data** or anything else: 

+ First a client downloads the model ***θ*** (which refers to the model parameters) from the server. 

+ The client trains the model locally by their own data and the distributed model ***θ***.

+ After the training is finished, the client will have a newly trained model ***θ'*** and ***(θ‘ - θ)***, which is also called "gradient difference" in deep learning process, will be uploaded to the server.

​  Therefore the client won't need to worry about the leakage of their image data, it is the "gradient difference" that is needed by the server rather than the image data. 

We've designed two schemes to upload the parameters to the server:

1. ***Full Automatically***: If all the processes run successfully and without any error, the parameters will be uploaded to the server after the training process is finished.

2. ***Self-automatically***: Otherwise, client could upload the parameters manually, and check if there are any newly aggregated parameters distributed by the server.

For fully automatic process, just run `python train.py   ` on the client side. We've integrated "register, download, train locally, upload" process in this file, which is convenient. Also, if there's some problems, you can execute these four process separately.   


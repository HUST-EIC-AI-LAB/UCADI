# COVID-19 Prediction With Federated Learning

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/3.0/"> Creative Commons Attribution-NonCommercial 3.0 Unported License.



## 1. Introduction

We provide a Federated Learning(FL) framework for researchers to train the model on remote machines.

Similar to most C/S structures, the framework consists of two parts: Server side, and Client side. To apply this framework in real scenarios, taking hospitals for example, the client part could be deployed on the machines of the hospital side (Client) where the federated model is trained locally, while the server part is established on the center machine (Server).

Once the scripts are executed, the hospitals will train their own models locally and transmit the parameters to the server which aggregates all parameters collected from the clients. Then the server distribute the newly aggregated parameters to each client of the FL process. This process iterate for some pre-set rounds before accuracy of the aggregated model reached the desired level.

Besides these, we add some useful functions to our frame work:

1. Encrypted parameters: Every Client could encrypt their model parameters trained locally via their generated private key. The Server will just aggregate these parameters but cannot decrypt them.
2. Weighted aggregation: We also add weighted aggregation method in this framework, researchers could assign different weight to each client for the final aggregation.



### 1.1 Communication settings

For the need of Encryption and Weighted Aggregation, it is not sufficient that Server and Clients only communicate the model parameters between them.

We define the file content format for this framework as follows:

​	File transmitted from Client contains:

> ​	"encrypted model_state": model encrpyted by the Client's private key
>
> ​	"client_weight": this Client's weight in the FL aggregation process

​	File transmitted from server contains: 

> ​	"model_state_dict": aggregated model parameters
>
> ​	"client_weight_sum": the weight_sum of Clients of  the current FL process
>
> ​	"client_num": the number of Clients of the current FL process

And we prepare ` pack/unpack functions` both in Server and Client class.

If one does not need the Encryption or Weighted Aggregation, the file format could be redefined.  All the files are stored in `.pth` format.



### 1.2 Server side

**1.2.1**   "Server folder" contains two main scripts `server_main.py` and `fl_server.py`. In `fl_server.py` we define the `FL_Server`  class, and in `server_main` we provide an example using `FL_Server` class.

**1.2.2**   Before starting the FL process, we need to set some configurations for the Server.

`./server/config/config.json`, you need to set the `.json` file for yourself:

> ```python
> {
>   "ip": "0.0.0.0",
>   "send_port": 12341,
>   "recv_port": 12342,
>   "clients_path": "./config/clients.json",
>   "model_path": "./model/model.py",
>   "weight_path": "./model/merge_model/initial.pth",
>   "merge_model_dir": "./model/merge_model/",
>   "client_weight_dir": "./model/client_model/",
>   "buff_size": 1024,
>   "iteration": 120
> }
> ```

`ip`,  `recv_port`: socket configurations.

`client_path`: `.json` file path to `clients.json`, which contains clients informations(`username` and `password`)

`weight_path`: the model path which will be sent to the clients in the FL process

`merge_model_dir`: after aggregations, merged model will be stored here.

`client_model_dir`: model_parameters(`.pth` file) received from each clients will be saved here, and after the aggregation, all files stored here will be removed.

`iterations`:  FL processing iterations num.



**1.2.3**  `./server/config/clients.json` stores the `username`  and `password`  of each Client.  Clients need to register to the server via these informations. If the information is wrong, the register request will be refused and cannot join the FL process.

There are some examples below:

> {
>
> ​	"Bob": "123456", 
>
> ​	"Alan": "123456", 
>
> ​	"John": "123456"
>
> }



### 1.3 Client side

**1.3.1**  `./client`  folder also contains 2 main scripts `fl_client.py`  in which we define the `FL_Client` class,  and `client1_main.py` which we provide an example.

**1.3.2**  Configurations of Clients are as below:

`./client/config/client1_config.json`

> ```python
> {
>   "username": "Bob",
>   "password": "123456",
>   "ip": "127.0.0.1",
>   "send_port": 12346,
>   "recv_port": 12347,
>   "server_ip": "127.0.0.1",
>   "server_port": 12342,
>   "buff_size": 1024,
>   "model_path": "./model/model.py",
>   "weight_path": "./model/initial.pth",
>   "models_dir": "./model",
>   "iteration": 120
> }
> ```

For the client, **pay attention** to its `FL_Client.train_model_path` attribute, which stores the path of the model parameter file received from the Server;  This attribute stores "weight_path" in the `client1_config.json`  when it is initialized, but the FL_Client class will change this value when the program is running,  the user can directly read the parameter file sent from the server through this path.



**1.3.3** Because the training process takes place on the client side, you also need to set you own train settings. Our configurations are given in the following as an example:

> ```python
> {
>   "train_data_dir": "/home/ABC/EFG/dataset/",
>   "train_df_csv": "./utils/HIJ.csv",
>   "labels_train_df_csv": "./utils/HIJ.csv",
>   "test_data_dir": "/home/ABC/EFG/dataset/",
>   "test_df_csv": "./utils/test_norm.csv",
>   "labels_test_df_csv": "./utils/test_norm.csv",
>   "use_cuda": true,
>   "epoch": 101,
>   "train_batch_size": 50,
>   "test_batch_size": 2,
>   "num_workers": 12,
>   "lr": 0.015,
>   "momentum": 0.9
> }
> ```



## 2.  FL framework installation

### Install FL framework from Github

You could run this command `git clone https://github.com/HUST-EIC-AI-LAB/COVID-19-Fedrated-Learinig.git` to deploy your own FL task.

#### Installation Dependencies

Some dependencies may need to be pre-installed, e.g. pytorch and CUDA, before you can train on GPU. Run `pip install -r requirement.txt` to install the required dependencies

**Notice:**
In `requirement.txt`, we use torch matches `cuda == 9.2.`

If there are problems in using torch, it may be caused by version mismatch between torch and CUDA, please check your CUDA version by `cat /usr/local/cuda/version.txt`, and download the correct version of Pytorch from the official website(https://pytorch.org/).

**ATTENTION:**

`ninja`  and `re2c`  are C ++ extension methods,  you should install them as described in their github.

`ninja` : https://github.com/ninja-build/ninja

`re2c`   : https://github.com/skvadrik/re2c

Our encryption algorithm comes from https://github.com/lucifer2859/Paillier-LWE-based-PHE



## 3. Implementation of FL

**3.1**   We have reduced the operations required in the communication process as much as possible. Yet, the Client-side training process and the Sever-side aggregating process still need to be customized by the researcher.

We provide a rough template for the communication process, `./server/server_main_raw.py`  and `./client/client1_main_row.py`. You can design your own federal learning process accordingly.

In addition, we also provide our federal learning code for COVID-19 prediction as an example, which contains encryption and weighted aggregation content.

First run the `server_main.py`  file, then run the `client1_main.py`  file on the client machine. You can add any number of clients, only need to modify the corresponding configuration file path in `client1_main.py`, such as: `client1_config.json`  and `train_config_client1.json`.

> ```python
> client = FL_Client('./config/client1_config.json')
> 
> client.start()
> 
> with open('./config/train_config_client1.json') as j:
>     train_config = json.load(j)
> ```

After modifying the corresponding configuration, run separately:

1. `python server_main.py`
2. `python client1_main.py`
3. `pyhton client2_main.py ......`



**3.2  Some tips**

​	Our FL process is designed to be more flexible. On the server side, you can select all registered clients to return to their parameter files and start aggregation. You can also set a minimum number of clients `min_clients` and a `maximum` delay: when enough number of clients return their files or no new client returns its file in the predefined maximum delay, the server can start the aggregation process in advance, and no longer receive any client requests.

 These contents can be found in the `server_main_raw.py` file and the corresponding code is provided.





## 4. Others

​	We have completed the communication between different devices in the local area network, and have done tests related to the Federated Learning process. Our communication process is based on Socket. If you want to successfully deploy this framework to different devices in different network domains, researchers may need to consider the corresponding port and firewall settings to ensure successful communication.

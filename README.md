## [UCADI] COVID-19 Diagnosis With Federated Learning 
<a rel="license" href="http://creativecommons.org/licenses/by-nc/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/3.0/88x31.png"/></a>
[![DOI](https://zenodo.org/badge/253738016.svg)](https://zenodo.org/badge/latestdoi/253738016)

## Intro

**We developed a Federated Learning (FL) Framework for global researchers to collaboratively train an AI diagnostic model based on various data centres without data sharing, in the name of UCADI, Unified CT-COVID AI Diagnostic Initiative.** 

We provide instructions for the effective deployment of UCADI in this manual.

Similar to prior structures, this framework consists of two parts: Server and Client. Applying this framework needs to set up a server first and has at least one client, which is ensured to ping the server successfully. Concretely, to train a federated model over various hospitals, a machine (a home PC is sufficient) is required to work in the cloud as the central server to collect, aggregate, and dispatch the encrypted model parameters of clients. Meanwhile, hospitals need more computation resources (usually a GPU workstation) and and sufficient internet bandwid to function as the clients.

Once the process starts, the hospitals will train their local models and transmit the encrypted parameters to the server. Then the server merges all parameter packets collected from the clients to update the global model, delivers the newly merged model parameters to each actively participating client. The FL process will be last for enough epochs until the federated model reaches the desired performance.

We equip the framework with some additional features:

1. **Homomorphic encryption**: each client is able to encrypt the parameters of their local trained model via the specified private key, and the server will aggregate those encrypted parameters without the ability to decrypt them;
2. **Weighted aggregation**: each client contributes the locally trained parameters with weight to the global federated model. The weight depends on the size of the dataset for training on the client.

#### Communication settings


For the need of encryption and weighted aggregation, it is not sufficient if the server and client only communicate the model parameters between them.

We define the file content format for this framework as follows:

​	File transmitted from Client contains:

> ​	"encrypted model_state": encrypted model parameters by the Client's private key
>
> ​	"client_weight": this Client's weight in the FL aggregation process (size of dataset for training)

​	File transmitted from Server contains: 

> ​	"model_state_dict": updated global model parameters
>
> ​	"client_weight_sum": the sum of clients' weight in the current FL process
>
> ​	"client_num": the number of clients in the current FL process

And we prepare ` pack_params/unpack_params`  functions both in Server and Client Class to generate/parse the file we mentioned above. If the encryption or weighted aggregation are not needed, via redefining the file format.  All the transmitted files are stored in `.pth` format.

### Server

`./server` folder contains two main scripts `server_main.py` and `fl_server.py`. In `fl_server.py` we define the `FL_Server`  class, and in `server_main.py` we provide an example using `FL_Server` class. Before starting the FL process, we need to set the server's configurations in `./server/config/config.json`.

> ```json
> {
>     "ip": "0.0.0.0",
>     "send_port": 12341,
>     "recv_port": 12342,
>     "clients_path": "./config/clients.json",
>     "model_path": "./model/model.py",
>     "weight_path": "./model/merge_model/initial.pth",
>     "merge_model_dir": "./model/merge_model/",
>     "client_weight_dir": "./model/client_model/",
>     "buff_size": 1024,
>     "iteration": 5
> }
> ```

`ip`,  `recv_port`: socket configurations, and `recv_port` listens constantly for the packets from clients.

`client_path`: `.json` file path to `clients.json`, which contains clients informations (`username` and `password`) which are accessible to join in the training process.

`weight_path`: the model which will be deliver to the clients in the FL process, and it will change automatically as the training process progresses. However, you need to define the initial path when you set up the serevr to start working.

`merge_model_dir`: each updated global model will be saved here.

`client_model_dir`: trained local model (`.pth` file) contributed by clients will be saved here, and the dir will be cleared when the aggregation process is over.

`iterations`:  rounds of FL training process.

`./server/config/clients.json` stores the `username`  and `password`  of each client.  Clients need to register to the server via this information. If the information is wrong, the register request will be refused and won't be allowed to participate in the FL process.

### Client

`./client`  folder contains two main scripts: In `fl_client.py`,  we define the `FL_Client` class,  and in `client_main.py`,  we provide an example of how to run it. The client's configurations are needed setting (template: ./client/config/client1_config.json`).

> ```json
> {
>   "username": "Bob",
>   "password": "123456",
>   "ip": "127.0.0.1",
>   "work_port": 12346,
>   "server_ip": "127.0.0.1",
>   "server_port": 12342,
>   "buff_size": 1024,
>   "model_path": "./model/model.py",
>   "weight_path": "./model/initial.pth",
>   "models_dir": "./model",
>   "seed": 1434
>   }
>   ```

`ip`: ip of client.

`work_port`: the free port in client to send packets to server and receive from server.

`server_ip` : ip of server.

`server_port`:   the port is opened by server to listen the sending request from client. same as the `recv_port` in server configuration.

`model_path`: the path to save the model architecture, which is also delivered by the central server.

 ` weight_path`: It record the start model in the current FL training process, and it will be set automatically as the training process progresses.

`model_dir`: the path to save the model weight trained locally.

`seed`: the seed is used to generate private key.

Because the training process also takes place on the Client machine, you also need to set your own train hyperparameter. Our configurations are given in the following as an example of `train_config_client.json'`:

> ```json
> {
>   "train_data_dir": "path to save raw data for training",
>   "train_df_csv": "csv file for training, and 'name' columns save the image name, 'label' columns svae the real label in classification task ",
>   "use_cuda": true,
>   "train_batch_size": 4,
>   "num_workers": 12,
>   "lr": 0.015,
>   "momentum": 0.9,
>   "iteration":1 // num of epochs to train locally.
>   }
>   ```



## Installation

**Install from GitHub:**

Developers could run this command `git clone https://github.com/HUST-EIC-AI-LAB/COVID-19-Federated-Learning.git` to deploy your own FL task.

**Dependencies：**

Some dependencies may need to be pre-installed, e.g. PyTorch and CUDA, before you can train on GPU. Run `pip install -r requirement.txt` to install the required dependencies

**Notice:**
In `requirement.txt`, we use `PyTorch` that matches `cuda == 9.2`.

If there are problems in using torch, it may be caused by version mismatch between torch and CUDA, please check your CUDA version by `cat /usr/local/cuda/version.txt` , and download the correct version of PyTorch from the official website.

**Attention:**

`ninja`  and `re2c`  are C ++ extension methods,  you should install them as described in their github.

`ninja` : https://github.com/ninja-build/ninja

`re2c`   : https://github.com/skvadrik/re2c

Our encryption algorithm comes from https://github.com/lucifer2859/Paillier-LWE-based-PHE

**Docker:**

we have also provide docker option, which supports `PyTorch 1.6.0` with `cuda 10.1`, where it automatically install the python dependencis in`requirements.txt` , `apx`, `ninja` and `re2c`. It is located in the `docker` folder.

To set up:

```bash
cd docker
sh build_docker.sh
sh launch_docker.sh
```



## Implementation

We have reduced the operations required in the communication process as much as possible. Yet, the Client training process and the Server aggregating process still need to be customized by the researcher.

We provide a rough template for the communication process, `./server/server_main_raw.py`  and `./client/client_main_raw.py`. Therefore you can design your own federated learning process accordingly.

In addition, we also provide our federated learning code for COVID-19 prediction as an example, which contains encryption and weighted aggregation as well.

To set up, first run the `server_main_raw.py`  file, then run the `client_main_raw.py`  file on the client machine. You can add any number of clients, only need to modify the corresponding configuration in `client_config.json`  and `train_config_client.json`.

> ```python
> // in client_main_raw.py
> client = FL_Client('./config/client_config.json')
> client.start()
> with open('./config/train_config_client.json') as j:
> 	train_config = json.load(j)
>    ```

After modifying the corresponding configuration, run each client worker separately:

```bash
# launch the central server
cd server && python server_main.py
# start training model on client 1
cd client 
CUDA_VISIBLE_DEVICES=0,1 python client1_main.py
# start training another model on client 2
CUDA_VISIBLE_DEVICES=2,3 python client2_main.py
# more clients can be added
```

**Some tips**

Our FL process has more flexibility. For the server, developers can select all registered clients to do aggregation. Or instead, you can also set a minimum number of clients `min_clients` and a maximum waiting time `timeout`. When enough clients finish transmitting or the time for clients to upload is running out (server starts timing when receiving the first packet from any client), the server will execute the aggregation process while no longer accept requests from any client. Meanwhile, the server will delete those clients not upload timely from the training group until they request to join in the training process again.

## Flow chart 

Our communication process is based on Web Socket. If you want to successfully deploy this framework in the real scenario, developers may need to consider the port setting and firewall settings to ensure the network connection is successful.

The flow chart is as following:

![](./pic/flow_chart.jpg)



## Citation 

If you find UCADI useful, please cite our tech report (outdated), a more recent draft is available upon request.

```bibtex
@article{ucadi,
title = {Advancing COVID-19 Diagnosis with Privacy-Preserving Collaboration in Artificial Intelligence},
author = {Bai, Xiang and Wang, Hanchen and Ma, Liya and Xu, Yongchao and Gan, Jiefeng and others},
year = 2021,
journal = {Nature Machine Intelligence},
publisher = {Nature Publishing Group}
}
```



## News

**[Sep 2021]:** We just get accepted by the Nature Machine Intelligence 🔥!

**[Jul 2021]:** We submitted the revised manuscript back to Nature Machine Intelligence, finger crossed!

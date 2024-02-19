## Docker Build (Optional)
*We further provide an optional Docker file which can be used to build working env with Docker.*

1.  Install docker to the system (postpone this step if docker has been already installed):
    ```shell
    sudo apt install docker
    ```
2. To start your docker environment, please go to the folder **dockerfiles**:
   ```shell
   cd dockerfiles
   ```
3. Create a docker image (you can put any name you want):
    ```shell
    docker build --tag 'laa_net' .
    ```
4. Check the status of the image created:
   1. Run command:
        ``` shell
        docker image ls
        ```
   2. You should see something similiar:
   
        |REPOSITORY| IMAGE ID|CREATED| SIZE |TAG|
        |---|---|---|---|---|
        |laa_net| efd422370750|12 minutes ago| 18.4GB |latest |
5. Run a container from the created image:
   1. Run command
        ```shell
        docker run -v <data_path>:<path_in_container> --gpus 'all,capabilities=utility' -it laa_net /bin/bash
        ```
   2. Check the container created:
       1. Run command:
            ```shell
            docker ps
            ```
       2. Check the result:
            CONTAINER ID|IMAGE|COMMAND|CREATED|STATUS|PORTS|NAMES|
            |---|---|---|---|---|---|---|
            |0203c192febb|laa_net|"/bin/bash"| 29 seconds ago |Up 28 seconds | |determined_cannon|
       3. To access the docker container:
            ```shell
            docker exec -it 0203c192febb /bin/bash
            ```
       4. To start the container:
            ```shell
            docker start 0203c192febb
            ```
       5. To stop the container:
            ```shell
            docker stop 0203c192febb
            ```

6. Inside the docker container, you can clone or mount the repository from outside:
    ```shell
    cd /workspace/
    git clone https://github.com/XXX/LAA-Net.git
    cd LAA-Net/
    ```
7. Now you are ready for [*QuickStart*](#quickstart)

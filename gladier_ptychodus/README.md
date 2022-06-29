# Ptychodus Gladier Client

## Running the client

## Creating the client

### Set up the environment

### Create the client file

We will create the repo by the [gladier-client-template](https://github.com/globus-gladier/gladier-client-template/tree/main/full_client), click in "Use this template" and clone the new repository.

Lets start by cleaning the files which will not be used.
```
mv README.md > GLADIER_JOURNEY.md
echo "# Ptychodus Gladier Client" > README.md

rm CITATION.cff
rm -rf simple_clients
```
Lets now rename the folder which will be used for our client
```
mv full_client gladier_ptychodus
cd gladier_ptychodus
mv full_client.py ptychodus_client.py
```

This is a base skeleton for a gladier client. Like many other scientific data workflows, this flow follows the pattern: **Transfer**, **Process**, **Publish** with small modifications. The full ptychodus flow has this steps:

1. **Transfer** Data from experiment
2. **Process** data with Ptychodus
3. **Process** result to create images
4. **Transfer** Data back to experiment
5. **Publish** Data and Images

Lets edit the flow now. 

The flow definition already have steps 1,2,3 and 5. We will ignore the missing transfer for now and work on the existing flow. 

```
class Example_Client(GladierBaseClient):
    gladier_tools = [
        TransferOut,
        SimpleTool,
        GatherMetadata,
        Publish
    ]
```
### Initial Transfer

The first transfer is from the experiment into the processing resource. We will use the already existing [Transfer Tool](https://github.com/globus-gladier/gladier-tools/blob/main/gladier_tools/globus/transfer.py) and use the `tool:prefix` syntax to differentiate it from the second transfer in the flow. The final line at the flow definition will then become:
```
"gladier_tools.globus.transfer.Transfer:FromStorage"
```
The transfer tool normally requires the following arguments
```
# globus local endpoint
"from_storage_transfer_source_endpoint_id": "80150e2e-5e88-4d35-b3cd-170b25b60538",
"from_storage_transfer_source_path": str(local_dir),
# eagle endpoint for first transfer (may be repeated since it is eagle/eagle)
"from_storage_transfer_destination_endpoint_id": "80150e2e-5e88-4d35-b3cd-170b25b60538",
"from_storage_transfer_destination_path": str(data_dir),
"from_storage_transfer_recursive": True,
```
Notice that they have the "from_storage_" prefix. Which means they will only be used by this tool unless specified otherwise on the workflow.


### Main Ptychodus tool

Ptychodus can be executed by one single command line at the processing resource, since this is a bash call we will use the already existing [shell_cmd](https://github.com/globus-gladier/gladier-tools/blob/main/gladier_tools/posix/shell_cmd.py) gladier tool. The final line at the flow definition will then become:
```
"gladier_tools.posix.shell_cmd.ShellCmdTool"
```

To match the requirements of this tool. We need to add the following variables to the payload: 

```
"args": f"ptychodus -f /eagle/APSDataAnalysis/PTYCHO{data_dir} -b -s ptychodus.ini > ptychodus.log",
"cwd": f"/eagle/APSDataAnalysis/PTYCHO{data_dir}",
"timeout": 180,
```

### Pytchodus Image Plotting tool

Ptychodus does not have a standard plotting interface yet and therefore we will need to create the plotting tool ourselves. This tool will also gather the metadata necessary to create the portal.

### 

### Create the necessary tools
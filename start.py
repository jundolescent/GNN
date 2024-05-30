import subprocess

models = ['GCN', 'GAT', 'GIN', 'GraphSAGE']

for model in models:
    command = f'python ./GNN.py --model {model}'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    process.wait()
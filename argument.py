import argparse


def argument():
    parser = argparse.ArgumentParser(description='Graph Neural Network Argument Parser')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT', 'GIN', 'Edge'],
                        help='Choose the GNN model to run (default: GCN)')
    args = parser.parse_args()
    if (args.model == 'GCN' or args.model == 'GraphSAGE' or args.model == 'GAT' or args.model == 'GIN'
            or args.model == 'Edge'):
        return args.model
    else:
        raise ValueError("Invalid model type. Choose from 'GCN', 'GraphSAGE', 'GAT', 'GIN', 'Edge'.")

def argument_ml():
    parser = argparse.ArgumentParser(description='Machine Learning Model Argument Parser')
    parser.add_argument('--model', type=str, default='LR', choices=['LR', 'RF', 'SVM', 'KNN', 'DT'],
                        help='Choose the ML model to run (default: LR)')
    args = parser.parse_args()
    if (args.model == 'LR' or args.model == 'RF' or args.model == 'SVM' or args.model == 'KNN'
            or args.model == 'DT'):
        return args.model
    else:
        raise ValueError("Invalid model type. Choose from 'LR', 'RF', 'SVM', 'KNN', 'DT'.")

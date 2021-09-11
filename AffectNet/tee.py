import sys
import argparse
from modelTrain import *




def main(model,weights=None, epochs=None):
    if epochs ==None:
        epochs = 30
    train_generator,validation_generator = genData('dataset', 'valSet')
    trainModel(model,train_generator,validation_generator,weights,epochs)
    print("Model trained :",model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--model', dest='model', type=str, help='Name of the model')
    parser.add_argument('--weights', dest='weights', type=str, help='Path to the file weights')
    parser.add_argument('--epochs', dest='epochs', type=str, help='Num of epochs')
    args = parser.parse_args()
    model = args.model
    weights = args.weights
    try:
        epochs = int(args.epochs)
    except:
        epochs = 30
    if weights==None:
        print('WEIGHTS NULL')
    main(model,weights,epochs)
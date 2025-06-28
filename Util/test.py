import os.path
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.CNN_classifier import*
from Util.data_preprocessing import *
from torch.utils.tensorboard import SummaryWriter


def evaluate_model_on_test(x_test_data, y_test_data, model_path, device):
    # Create DataLoader for test data
    test_loader = DataLoader(TensorDataset(x_test_data, y_test_data), batch_size=1, shuffle=False)

    # Load model
    checkpoint = torch.load(os.path.join(model_path, 'model_full.pth'), map_location=device)
    model = checkpoint['model']
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, target = torch.max(labels.data, 1)

            correct += torch.sum(predicted == target).item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = test_loss / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # test model without mixup
    print('Model score without mix up')
    best_model_dir = 'best_model'
    data = DATA('../data', wav_to_pkl=False, with_mixup=False)
    evaluate_model_on_test(data.X_test, data.y_test, '../' + best_model_dir, device)
    evaluate_model_on_test(data.X_test_with_mixup, data.y_test_with_mixup, '../' + best_model_dir, device)

    # test model with mixup
    print('\nModel score with mix up')
    best_model_dir = 'best_model_with_mixup'
    data = DATA('../data', wav_to_pkl=False, with_mixup=True)
    evaluate_model_on_test(data.X_test, data.y_test, '../' + best_model_dir, device)
    evaluate_model_on_test(data.X_test_with_mixup, data.y_test_with_mixup, '../' + best_model_dir, device)
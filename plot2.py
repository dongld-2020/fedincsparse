import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_files):
    """
    Vẽ biểu đồ so sánh các chỉ số từ các file CSV.
    :param csv_files: Danh sách đường dẫn đến file CSV
    """
    # Danh sách để lưu dữ liệu từ các file
    all_data = []
    algorithms = []
    training_params = None

    # Đọc dữ liệu từ từng file CSV và trích xuất thông số huấn luyện
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found!")
            continue
        df = pd.read_csv(csv_file)
        # Trích xuất tên thuật toán từ tên file
        filename_parts = csv_file.split('_')
        algorithm = filename_parts[1]  # Ví dụ: fedavg, fedprox, scaffold
        algorithms.append(algorithm.capitalize())

        # Trích xuất các thông số huấn luyện từ tên file
        params = {
            'clients': filename_parts[2].replace('clients', ''),  # NUM_CLIENTS
            'rounds': filename_parts[3].replace('rounds', ''),  # NUM_ROUNDS
            'clientsPerRound': filename_parts[4].replace('clientsPerRound', ''),  # NUM_CLIENTS_PER_ROUND
            'epochs': filename_parts[5].replace('epochs', ''),  # LOCAL_EPOCHS
            'alpha': filename_parts[6].replace('alpha', ''),  # ALPHA
            'lr': filename_parts[7].replace('lr', ''),  # LEARNING_RATE
            'seed': filename_parts[8].replace('seed', '').replace('.csv', '')  # GLOBAL_SEED
        }

        # Kiểm tra tính nhất quán của các thông số huấn luyện giữa các file
        if training_params is None:
            training_params = params
        else:
            if training_params != params:
                print(f"Warning: Training parameters in {csv_file} do not match previous files!")
                print(f"Expected: {training_params}")
                print(f"Got: {params}")

        all_data.append((algorithm, df))

    if not all_data:
        print("No valid CSV files to plot!")
        return

    # Tạo chuỗi thông số huấn luyện để thêm vào tên file PNG
    params_str = (f"clients{training_params['clients']}_rounds{training_params['rounds']}_"
                 f"clientsPerRound{training_params['clientsPerRound']}_epochs{training_params['epochs']}_"
                 f"alpha{training_params['alpha']}_lr{training_params['lr']}_seed{training_params['seed']}")

    # Tạo tiêu đề với danh sách các thuật toán được so sánh
    algorithms_str = ", ".join(algorithms)

    # 1. Vẽ biểu đồ Loss và Accuracy trên cùng 1 figure (2 subplot ngang hàng)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1 hàng, 2 cột

    # Vẽ biểu đồ Loss trên ax1
    for algorithm, df in all_data:
        if algorithm != 'fedsparse':            
            ax1.plot(df['Round'], df['Loss'], label=f'{algorithm.capitalize()} Loss', linestyle='--', linewidth=2)
        else:
            ax1.plot(df['Round'], df['Loss'], label=f'FedSparse Loss', linewidth=3)
            
    ax1.set_title(f'Validation Loss Over Rounds Comparing', fontsize=20)
    ax1.set_xlabel('Communication Round', fontsize=20)
    ax1.set_ylabel('Validation Loss', fontsize=20)
    ax1.legend(fontsize=20)
    ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=20)  # Tăng font size cho số trên cả hai trục

    # Vẽ biểu đồ Accuracy trên ax2
    for algorithm, df in all_data:
        if algorithm != 'fedsparse':
            ax2.plot(df['Round'], df['Accuracy'], label=f'{algorithm.capitalize()} Accuracy', linestyle='--', linewidth=2)
        else:
            ax2.plot(df['Round'], df['Accuracy'], label=f'FedSparse Accuracy', linewidth=3)
    ax2.set_title(f'Accuracy Over Rounds Comparing', fontsize=20)
    ax2.set_xlabel('Communication Round', fontsize=20)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=20)
    ax2.legend(fontsize=20)
    ax2.grid(True)
    ax2.tick_params(axis='both', labelsize=20)  # Tăng font size cho số trên cả hai trục
    # Lưu và hiển thị hình ảnh
    comparison_filename = f'comparison_{params_str}.png'
    plt.tight_layout()  # Đảm bảo các biểu đồ không bị chồng lấp
    plt.savefig(comparison_filename)
    plt.show()
    plt.close()
    print(f"Saved {comparison_filename}")

if __name__ == "__main__":
    # Yêu cầu người dùng nhập tên file CSV
    print("Enter the CSV files to compare (e.g., results_fedavg_..., results_fedprox_...).")
    print("Separate multiple files with a comma (,). Press Enter to finish.")
    csv_input = input("CSV files: ").strip()
    
    if not csv_input:
        print("No files provided! Please run again and enter at least one CSV file.")
    else:
        csv_files = [f.strip() for f in csv_input.split(',')]
        plot_metrics(csv_files)

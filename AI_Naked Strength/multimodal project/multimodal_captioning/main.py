from utils.data_loader import get_loader

def main():
    csv_file = 'TBD'
    img_dir = 'TBD'


    data_loader = get_loader(
        csv_file = csv_file,
        img_dir = img_dir,
        batch_size = 32,
        shuffle=True,
        num_workers=4
    )

    # 데이터 확인
    for images, captions in data_loader:
        print("Image batch shape:", images.shape)
        print("Sample caption:", captions[0])
        break


if __name__ == "__main__":
    main()
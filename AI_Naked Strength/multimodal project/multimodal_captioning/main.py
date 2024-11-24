from utils.data_loader import get_loader

def main():
    img_dir = 'AI_Naked Strength/multimodal project/data/flickr30k/images'
    caption_file = 'AI_Naked Strength/multimodal project/data/flickr30k/captions.txt'

    data_loader = get_loader(
        img_dir = img_dir,
        captino_file=caption_file,
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
from Trainer.dataset import TextFile


file = TextFile(
                file_path='Trainer/data.txt', 
                batch_size=16,
                token_count=1024, 
                ddp_world_size= 4,
                ddp_rank=0, device='mps'
                )
while True:
    try: 
        x = file.next_batch()
        print(file.current_pos)
    except:
        break
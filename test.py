import random
from data import ImageDetectionsField, TextField, RawField
from data import Sydney, UCM, RSICD,NWPU, DataLoader
from models.transformer.mamba_lm import MambaLM, MambaLMConfig
import evaluation
from models.transformer import Transformer, VisualEncoder, MeshedDecoder, ScaledDotProductAttention
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import warnings
import json
warnings.filterwarnings("ignore")


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch()


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)



    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='RSIC-GMamba')
    parser.add_argument('--exp_name', type=str, default='NWPU')# Sydney UCM RSICD NWPU
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    ################################################################################################################
    # parser.add_argument('--annotation_folder', type=str,
    #                     default='/media/dmd/ours/mlw/rs/UCM_Captions')
    # parser.add_argument('--features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/clip_feature/ucm_224')
    ################################################################################################################
    # parser.add_argument('--annotation_folder', type=str,
    #                     default='/media/dmd/ours/mlw/rs/RSICD_Captions')
    # parser.add_argument('--features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/clip_feature/rsicd_224')
    # parser.add_argument('--checkpoint', type=str,
    #                     default='./saved_models/nwpu/RSICD_best.pth')
    ################################################################################################################
    parser.add_argument('--annotation_folder', type=str,
                        default='/media/dmd/ours/mlw/rs/NWPU_Captions')
    parser.add_argument('--features_path', type=str,
                        default='/media/dmd/ours/mlw/rs/clip_feature/nwpu_224')
    ################################################################################################################

    args = parser.parse_args()

    print('Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    if args.exp_name == 'Sydney':
        dataset = Sydney(image_field, text_field, 'Sydney/images/', args.annotation_folder)
    elif args.exp_name == 'UCM':
        dataset = UCM(image_field, text_field, 'UCM/images/', args.annotation_folder)
    elif args.exp_name == 'RSICD':
        dataset = RSICD(image_field, text_field, 'RSICD/images/', args.annotation_folder)
    elif args.exp_name == 'NWPU':
        dataset = NWPU(image_field, text_field, 'NWPU/images/', args.annotation_folder)


    _, _, test_dataset = dataset.splits

    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    
    encoder = VisualEncoder(3, 0, attention_module=ScaledDotProductAttention)
    decoder = MambaLM(lm_config=MambaLMConfig, vocab_size=len(text_field.vocab), max_len=127,
                      padding_idx=text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

   
    data = torch.load('./saved_models/NWPU_best.pth')

    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores['BLEU'][0])
    print(scores['BLEU'][1])
    print(scores['BLEU'][2])
    print(scores['BLEU'][3])
    print(scores['METEOR'])
    print(scores['ROUGE'])
    print(scores['CIDEr'])
    print(scores['SPICE'])


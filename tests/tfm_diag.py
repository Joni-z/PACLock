import sys, os, torch
V = "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/vendor/tfm"
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
sys.path.insert(0, V)
from models.tfm_token import get_tfm_tokenizer_2x2x8, get_tfm_token_classifier_64x4
from paclock_bench.models.foundation.tfm_adapter import build_tfm, weight_paths
W = V + "/pretrained_weigths"

def cmp(model, sd, tag):
    ms, ws = model.state_dict(), sd
    mk, wk = set(ms), set(ws)
    print("## %s: model=%d weight=%d common=%d" % (tag, len(mk), len(wk), len(mk & wk)))
    if mk - wk: print("   model-only :", sorted(mk - wk)[:10])
    if wk - mk: print("   weight-only:", sorted(wk - mk)[:10])
    for k in sorted(mk & wk):
        if tuple(ms[k].shape) != tuple(ws[k].shape):
            print("   SHAPE %s model=%s weight=%s" % (k, tuple(ms[k].shape), tuple(ws[k].shape)))

for tag, path, mk in [
    ("tokenizer single TUAB", W + "/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth",
     lambda: get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)),
    ("tokenizer multi", W + "/multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth",
     lambda: get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)),
    ("encoder single TUAB", W + "/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth",
     lambda: get_tfm_token_classifier_64x4(n_classes=2, code_book_size=8192, emb_size=64)),
    ("encoder multi MTP", W + "/multiple_dataset_settings/MTP_Pretrained_tfm_encoder_64x4/tfm_encoder_mtp_last.pth",
     lambda: get_tfm_token_classifier_64x4(n_classes=2, code_book_size=8192, emb_size=64)),
]:
    if not os.path.exists(path):
        print("## %s: MISSING %s" % (tag, path)); continue
    cmp(mk(), torch.load(path, map_location="cpu", weights_only=False), tag)

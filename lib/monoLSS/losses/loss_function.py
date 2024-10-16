from lib.monoLSS.helpers.decode_helper import _transpose_and_gather_feat

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C

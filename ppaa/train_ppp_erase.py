import os 
import os.path as osp
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from collections import defaultdict
import torch
import random
import numpy as np
from safetensors.torch import load_file
from typing import Union

import math
import sys
# os.environ["PYTHONHASHSEED"] = str(123)s
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import argparse
sys.path.append('.')
from utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call

# from utils.gradient_surgery import collect_param_grads,zero_param_grads,param_grad_stats,generalize_gradient_projection,inject_resolved_grads_by_name
# from utils.gradient_surgery import do_grad_injection
# from utils.gradient_surgery import generalize_gradient_projection_prob
from diffusers import DDIMScheduler
import wandb 
import torch
import torch.nn.functional as F
from copy import deepcopy
#+++ Constant

# these concepts do not have preservation concepts (yet)

celebs_50train = ["Bill Clinton", "Jared Leto", "Shirley Temple", "Shia LaBeouf", "Zayn Malik",
         "George Clooney", "Jennifer Lawrence", "Ronald Reagan", "Anjelica Huston", "Charlie Sheen",
         "Meryl Streep", "Angelina Jolie", "Bruce Willis", "Drake", "Demi Lovato",
         "Barack Obama", "Whoopi Goldberg", "Philip Seymour Hoffman", "Margot Robbie", "Nick Jonas",
         "Glenn Close", "Emma Stone", "Olivia Wilde", "Ryan Gosling", "Richard Gere",
         "Tom Hanks", "Jennifer Aniston", "Tom Hiddleston", "Jennifer Lopez", "Justin Timberlake",
         "Drew Barrymore", "Idris Elba", "Kate Winslet", "Adriana Lima", "Jake Gyllenhaal",
         "Frida Kahlo", "Ricky Gervais", "Jessica Biel", "Hugh Jackman", "Charlize Theron",
         "Beth Behrs", "Reese Witherspoon", "Johnny Depp", "Chris Hemsworth", "Mick Jagger",
         "Jason Momoa", "Andrew Garfield", "Jessica Chastain", "Courteney Cox", "Spike Lee"]
# artists_50train = ["Brice Marden", "Casey Weldon", "Christopher Balaskas", "Christophe Vacher", "Clemens Ascher",
#          "Carl Barks", "Catrin Welz-Stein", "Chris Ware", "Brian K. Vaughan", "Bruce Nauman",
#          "Charles Vess", "Brian Froud", "Brooke Shaden", "C. R. W. Nevinson", "Butcher Billy",
#          "Brian Sum", "Claude Cahun", "Chris Moore", "Charles Schridde", "Chen Zhen",
#          "Carl Gustav Carus", "Candido Portinari", "Chip Zdarsky", "Christian Dimitrov", "Chris Saunders",
#          "Clara Miller Burd", "Cassius Marcellus Coolidge", "Clarence Holbrook Carter", "Cedric Peyravernay", "Charles E. Burchfield",
#          "Cagnaccio Di San Pietro", "Carlos Schwabe", "Charles Le Brun", "Brett Weston", "Carmen Saldana",
#          "Carel Willink", "Chris Turnham", "Chantal Joffe", "Carlo Crivelli", "Bruce Pennington",
#          "Briana Mora", "Chris Ofili", "Charles Camoin", "Bruno Catalano", "Charles-Francois Daubigny",
#          "Caspar David Friedrich", "Brian Despain", "Charles Addams", "Bryan Hitch", "Christopher Jin Baron"]


artists_50train = [
'Van Gogh', 'Claude Monet', 'Picasso', 'Jackson Pollock', 'Akira Toriyama',
'George Stubbs', 'Carne Griffiths', 'Alfred Sisley', 'Charles Hermans', 'Pieter Bruegel The Elder',

'Aaron Horkey', 'Frank Bowling', 'Brian Mashburn', 'Amedeo Modigliani', 'Edouard Cortes',
'Fernand Khnopff', 'Jean Baptiste Simeon Chardin', 'Benozzo Gozzoli', 'Jehan Georges Vibert', 'Henri De Toulouse Lautrec',

'William Turner', 'Alex Garant', 'Pieter De Hooch', 'Rosalyn Drexler', 'William James Glackens',
'Charles Blackman', 'Leroy Neiman', 'Frans Hals', 'Pierre Auguste Renoir', 'Brian Kesinger',

'Jay Defeo', 'N.C. Wyeth', 'Franz Xaver Winterhalter', 'Mary Cassatt', 'Gustav Klimt',
'Edmund Charles Tarbell', 'Paul Delvaux', 'Arthur Hughes', 'Chiharu Shiota', 'Charline Von Heyl',

'Romare Bearden', 'Egon Schiele', 'Thomas Moran', 'Georges Seurat', 'Casey Weldon',
'Hans Hofmann', 'Pietro Perugino', 'Joan Miro', 'Alberto Vargas', 'Edwin Henry Landseer'
]
concept2prompt_templates = {
    'object': [
        "{}",
        "a photo of {}",
        "a photo of a {}",
        "an image of {}",
        "a picture of {}",
    ],
    'style':   [ 
    "a painting in the style of {}",
    "an artwork of {}",
    "a photo in the style of {}",
    "a picture in the style of {}",]
}

concept2preserve_prompt_templates = deepcopy(concept2prompt_templates)

Celeb_Prelim_List = ["Margot Robbie", "Anne Hathaway", "Amy Adams", "Emma Stone", "Amber Heard",
                     "David Beckham", "Chris Hemsworth", "Elon Musk", "Adam Driver", "Andrew Garfield",
                     "Barack Obama", "Morgan Freeman", "Chris Rock", "Will Smith", "Idris Elba",
                     "Rihanna", "Oprah Winfrey", "Zendaya", "Nicki Minaj",  "Octavia Spencer"
                    ]

Artistic_Prelim_List = [
    "a painting in the style of Van Gogh",
    "a painting in the style of Claude Monet",
    "a painting in the style of Picasso",
    "a painting in the style of Jackson Pollock",
    "a painting in the style of Salvador Dalí"
]


concept2multiple_concepts = {
    "4CELEB00": ["Margot Robbie", "David Beckham", "Barack Obama", "Rihanna"],
    "8CELEB00": ["Margot Robbie", "David Beckham", "Barack Obama", "Rihanna"] + ["Emma Stone","Elon Musk","Morgan Freeman","Oprah Winfrey"],
    "4ARTIST00": ["a painting in the style of Van Gogh", "a painting in the style of Claude Monet", "a painting in the style of Picasso", "a painting in the style of Jackson Pollock"],
    "9NAKED00": [f"{a} {b}" for a in ["naked","nude","erotic"] for b in ["person","man","woman"]],
    "3PNAKED00":  ["naked","nude","erotic"],

}
concept2multiple_concepts['50CELEB00'] = celebs_50train
concept2multiple_concepts['50ARTIST00'] = [f'a painting in the style of {artist}' for artist in artists_50train]


concept2multiple_generic_concepts = {
    "9NAKED00": [f"{a} {b}" for a in ["dressed","dressed","dressed"] for b in ["person","man","woman"]],

}

bypass_preservation_concepts = [
    "naked person",
    "naked woman",
    "naked man",
    "mackerel tabby cat",
    "beagle dog",
    "a painting in the style of Picasso",
    "Jesus Christ",
    "ipad",
    "macbook",
    "poodle dog"
]
bypass_preservation_concepts += Celeb_Prelim_List
bypass_preservation_concepts += Artistic_Prelim_List

concept2shortname = {
    "Margot Robbie": "mrobbie",
    "David Beckham": "beckham",
    "Barack Obama": "obama",
    "Rihanna": "rihanna",
    
    
    "Chris Hemsworth": "chemsworth",
    "Adam Driver": "adriver",
    "Andrew Garfield": "agarfield",
    "Anne Hathaway": "ahathaway",
    "Amy Adams": "aadam",
    "Emma Stone": "estone",
    "Amber Heard": "aheard",
    "Morgan Freeman": "mfreeman",
    "Chris Rock": "crock",
    "Will Smith": "willsmith",
    "Idris Elba": "ielba",
    "Oprah Winfrey": "owinfrey",
    "Elon Musk": "elon",
    "Zendaya": "zendaya",
    "Nicki Minaj": "nminaj",
    "Octavia Spencer": "ospencer",
            
    "pad thai": "padthai",
    "Donald Trump": "dtrump",
    "persian cat": "percat",
    "mackerel tabby cat": "maccat",
    "beagle dog": "bdog",
    "poodle dog": "pddog",
    
    "English Springer": "espring",
    
    "ganesha": "ganesha",
    "tank": "tank",
    "a painting in the style of Van Gogh": "vgogh",
    "a painting in the style of Claude Monet": "cmonet",
    "a painting in the style of Picasso": "picasso",
    "a painting in the style of Jackson Pollock": "pollock",
    "a painting in the style of Salvador Dalí": "dali",
    
    "a painting in the style of Akira Toriyama":'toriyama',
    "a painting in the style of Georges Seurat":'gsrat',
    "a painting in the style of Joan Miro":'jmiro',
    "a painting in the style of Egon Schiele": 'egon',
    # NSFW concepts 

    "naked": "pnaked",
    "naked person": "naked",
    "naked woman": "nakedw",
    "naked man": "nakedm",
    
    "Jesus Christ": "jesus",
    "ipad": "ipad",
    
    
    "Mickey Mouse": "mmouse",
    "Grumpy Cat": "gcat",
    # "R2D2 robot": "r2d2", # ECCV
    "R2D2": "r2d2",
    
    "Macbook": "macbook",
    
    
    "Snoopy": "snoopy",
    "SpongeBob": "spbob",
    "Goku": "goku",
    "Mario": "mario",
    
    
    "Venice": "venice",
    "Shibuya": "shibuya",
    "Iguazu waterfall": "iguazu",
    "a painting in the style of Cyberpunk": "cyberpunk",

    
    
}

# seed = 123
# rng = np.random.RandomState(seed=seed)

concept2generic_concept = {
                            "pad thai": "food dish",
                            "Donald Trump": "person",
                            "persian cat": "cat",
                            "mackerel tabby cat": "cat",
                            "beagle dog": "dog",
                            "poodle dog": "dog",
                            "English Springer": "dog",
                            
                            
                            "ganesha": "statue",
                            "tank": "car",
                            
                            "a painting in the style of Van Gogh": "a painting in the style of artist",
                            "a painting in the style of Claude Monet": "a painting in the style of artist",
                            "a painting in the style of Picasso": "a painting in the style of artist",
                            "a painting in the style of Jackson Pollock": "a painting in the style of artist",
                            "a painting in the style of Salvador Dalí": "a painting in the style of artist",
                     
                            "a painting in the style of Akira Toriyama":"a painting in the style of artist",
                            "a painting in the style of Georges Seurat":"a painting in the style of artist",
                            "a painting in the style of Joan Miro":"a painting in the style of artist",
                            "a painting in the style of Egon Schiele":"a painting in the style of artist",
                            
                            
                            
                            "naked": "dressed",
                            "naked person": "dressed person",
                            "naked woman": "dressed woman",
                            "naked man": "dressed man",
                            
                            "ipad": "tablet",
                            "Jesus Christ": "god",
                            
                            
                            
                            "Margot Robbie": "person",
                            # "David Beckham": "the person", #hack
                            # "David Beckham": "person person",
                            "David Beckham": "person",
                            "Barack Obama": "person",
                            "Rihanna": "person",
    
                            "Chris Hemsworth": "person",
                            "Adam Driver": "person",
                            "Andrew Garfield": "person",
                            "Anne Hathaway": "person",
                            "Amy Adams": "person",
                            "Emma Stone": "person",
                            "Amber Heard": "person",
                            "Morgan Freeman": "person",
                            "Chris Rock": "person",
                            "Will Smith": "person",
                            "Idris Elba": "person",
                            "Oprah Winfrey": "person",
                            "Elon Musk": "person",
                            "Zendaya": "person",
                            "Nicki Minaj": "person",
                            "Octavia Spencer": "person",
                            
                            
                            "Mickey Mouse": "cartoon",
                            # "R2D2 robot": "robot", # ECCV
                            "R2D2": "robot", # ECCV
                            
                            "Grumpy Cat": "cat",
                            "Macbook": "laptop",
                            
                            "4CELEB00" : "person",
                            "8CELEB00" : "person",
                            "50CELEB00" : "person",
                            "4ARTIST00" : "a painting in the style of artist",
                            
                            "50ARTIST00" : "a painting in the style of artist",
                            
                            "9NAKED00": "dressed person",
                            "3PNAKED00" : "dressed",  # ["nake"naked man","naked woman","nude person", "nude woman", "nude man"],
                            
                            "Mario": "game character",
                            "Snoopy": "cartoon character",
                            "SpongeBob": "cartoon character",
                            "Goku": "anime character",
                            
                            "Venice": "city",
                            "Shibuya": "city",
                            "Iguazu waterfall": "waterfall",
                            "a painting in the style of Cyberpunk": "a painting in the style of artist",


                            
                            
                            }
concept2SpecialEval = {
    "Margot Robbie": ["Harley Quinn", "Margot Robbie on a horse", "Margot Robbie in a movie scene"],
    "mickey mouse": ["mickey mouse in disneyland", "mouse in a style of cartoon", "cartoon mouse character", "mouse in disney style", "mouse"],
    "Barack Obama": ["the 44th president of the united states", "Barack Obama in disneyland", "Barack Obama in a style of cartoon", "the black president of the united states"],
    "Donald Trump": ["the 45th president of the united states", "the 45th president of the united states in disneyland","Donald Trump in disneyland", "Donald Trump in a style of cartoon",],
    "persian cat": ["white persian cat", "persian man", "dog", "cat"],
    "mackerel tabby cat": ["tabby cat with stripes", "striped cat", "mixed breed cat", "persian cat", "dog", "cat"],
    "beagle dog": ["beagle puppy", "beagle dog running", "golden retriever dog", "persian cat", "cat", "dog"],
    "poodle dog": ["beagle puppy", "beagle dog running", "golden retriever dog", "persian cat", "cat", "dog"] +  ["bichon frise dog","poodle dog","afghan hound dog"]+["greyhound dog","dalmatian dog","mexican hairless dog"],
    
    "pad thai": ["pad thai", "noodles with shrimp and tofu", "spagetti dish", "spagetti with tomato sauce", "fried rice with vegetables"],
    "tank": ["green car"],
    "ganesha": ["statue of a god","elephant statue",'hindu god statue'],
    
    
    "a painting in the style of Van Gogh": ["a painting in the style of Van Gogh", "a painting in the style of Claude Monet", "a starry night painting", "a sunflower painting"],
    "a painting in the style of Claude Monet": ["a painting in the style of Claude Monet", "a painting in the style of Van Gogh", "a water lilies painting", "a haystacks painting"],
    "a painting in the style of Picasso": ["a painting in the style of Picasso", "a cubism painting", "a surrealism painting", "a modern art painting","a painting in the style of Van Gogh", "a painting in the style of Claude Monet"],
    
    "naked person": ["naked man","naked woman","dressed person","dressed man","dressed woman"],
    
    "macbook": ["dell laptop", "hp laptop", "lenovo laptop", "asus laptop", "desktop computer"],
    "ipad": ["samsung tablet", "amazon tablet", "lenovo tablet", "microsoft tablet", "smartphone"],
    "Jesus Christ": ["Buddha", "Muhammad", "Moses", "Krishna", "Zeus"]
}

concept2neighbor = {
    "a painting in the style of Picasso": "a painting in the style of Claude Monet",
    "a painting in the style of Van Gogh": "a painting in the style of Claude Monet",
    "mackerel tabby cat": "persian cat",
    "beagle dog": "corgi dog",
    "Jesus Christ": "Buddha",
    "ipad": "samsung tablet",
    "macbook": "dell laptop",
    
}
#---

#+++ Utility_Functions


def number_to_scientific_str(x: float) -> str:
    """
    Convert a positive float to compact scientific notation (e.g., 1 -> '1e0',
    0.05 -> '5e-2', 0.0001 -> '1e-4').
    Assumes x > 0.
    """
    if x == 0:
        return "0e0"

    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)

    # Clean mantissa: remove trailing .0
    if mant.is_integer():
        mant = int(mant)

    return f"{mant}e{exp}"

# my own implementation
def load_token_embedding(text_encoder, tokenizer, weight_path):
    print(f"Loading Token Embeddings from {weight_path}")
    # Load the saved token embeddings
    loaded_embeds_dict = torch.load(weight_path)
    # Get the input embedding layer
    token_embeddings = text_encoder.get_input_embeddings()
    # Process each token
    for token, embed in loaded_embeds_dict.items():
        # Check if token already exists in tokenizer
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f'adding {token} to the tokenizer vocabs')
            # Token doesn't exist, add to tokenizer
            tokenizer.add_tokens([token])
            token_id = tokenizer.convert_tokens_to_ids(token)
            # Resize the embedding layer to match new vocab size
            text_encoder.resize_token_embeddings(len(tokenizer))
        # Set the embedding weight
        with torch.no_grad():
            print(f'loading embedding for {token}')
            token_embeddings.weight[token_id] = embed.to(token_embeddings.weight.device)
            
            

def _value_based_probs_divmax(timesteps: torch.Tensor, alpha: float, is_inverse=False) -> np.ndarray:
    """
    Compute bias directly from timestep values.
    Normalize by dividing by max(t). This ensures scale invariance and simplicity.

    We map lower timesteps → higher probability:
        s = 1 - (t / t_max)
        p ∝ s^alpha
    """
    t = torch.tensor(timesteps)
    t_max = torch.max(t)
    if t_max <= 0:
        probs = torch.ones_like(t) / t.numel()
        return probs.cpu().numpy()

    if is_inverse:
        s = t / t_max      # 0 at smallest (earliest), 1 at largest (latest)
    else:
        s = 1.0 - (t / t_max)     # 0 at largest (earliest), 1 at smallest (latest)
    s = s.clamp(0.0, 1.0)
    probs = torch.pow(s, alpha)
    if torch.sum(probs) <= 0:
        probs = torch.ones_like(probs)
    probs = probs / probs.sum()
    return probs.cpu().numpy()



def resolve_model_name(args): #, training_step):
    erase_concept = args.erase_concept
    train_method = args.train_method
    erase_concept_shortname = concept2shortname[erase_concept] if erase_concept in concept2shortname else erase_concept.replace(' ', '-')
    # base_file_name = f"{train_method}.{erase_concept_shortname}"


    base_file_name = f"{train_method}"
    if args.negative_guidance:
        base_file_name += f".nG{args.negative_guidance:.2f}"
        
        if args.neg_guidance_method != 'cfg':
            base_file_name += f"-{args.neg_guidance_method}"
        
        
    if args.extra_forward_prob is not None and args.extra_forward_prob > 0 and args.forward_general:
        
        if args.use_indiv_extra_forward:
            base_file_name += f".iFW" # indiv extra forward prob
        else:
            base_file_name += f".FW" # extra forward prob
        
        if args.forward_general:
            base_file_name += f"g" # extra forward general
        
        if args.forward_preserve:
            base_file_name += f"p" # extra forward preserve
        
        base_file_name += f"{args.extra_forward_prob:.2f}"
        

        if args.extra_forward_negative_guidance is  not None:
            if args.extra_forward_negative_guidance == 0.0:
                base_file_name += ".zg"
            
    
    if args.base_concept == 'general':
        base_file_name += f".bG" # base general
    elif args.base_concept == 'neighbor':
        base_file_name += f".bN" # base neighbor
    
    if args.erase_from is not None:
        if args.erase_from == 'uncond':
            base_file_name += f".fU" # erase from uncond
        elif args.erase_from == 'object':
            base_file_name += f".fO" # erase from object    
            
        elif args.erase_from == 'general':
            base_file_name += f".fG" # erase from general    
        
        elif args.erase_from == 'forward':
            base_file_name += f".fF" # erase from forward
            
        elif args.erase_from == 'neighbor':
            base_file_name += f".fN" # erase from neighbor 
         
         
    if args.timestep_sampler is not None:
        if 'alpha' in args.timestep_sampler:
           timestep_sampler = args.timestep_sampler.replace('alpha','a.')
        base_file_name += f".TS{timestep_sampler}"
    elif args.timestep_constraint is not None:
        base_file_name += f".T{args.timestep_constraint}"
        
           
            

    if not args.apply_gradient_projection and args.preservation_weight is not None and args.preservation_weight > 0:
        base_file_name += '.'
        if  args.preservation_train_set:
            if args.preservation_train_set in ['00','01','02','03'] + ['G','UG','U']:
                base_file_name += f"pe{args.preservation_train_set}"
            elif args.preservation_train_set == 'celeb':
                base_file_name += f"cl"
            elif args.preservation_train_set == 'coco':
                base_file_name += f"cc"
            if args.preservation_split != 'train':
                base_file_name += f".{args.preservation_split.upper()}"
            base_file_name += '-'

                
        
        if args.preservation_weight_option == 'convex' and args.preservation_weight != 0.0:
            base_file_name += f"cPS{args.preservation_weight:.2f}"
        else:
            base_file_name += f"PS{args.preservation_weight:.2f}"
        
        
    if args.aei_loss_weight is not None and args.aei_loss_weight > 0.0:
        base_file_name += f"_{args.aei_loss_weight:.2f}A"
        if args.sim_param_group == 'attn_head':
            base_file_name += f"h"
        if args.sim_param_group == 'token':
            base_file_name += f"t"
        if args.sim_param_group == 'average_token':
            base_file_name += f"a" 
            
        if args.ang_aggr_feature_option != 'avg':
            base_file_name += f"{args.ang_aggr_feature_option[0].capitalize()}"
            
        if args.use_l2_loss:
            base_file_name += 'L2'
            
        # base_file_name += f"{args.aei_loss_weight:.2f}"
        
        if args.ang_excl_margin is not None:
            base_file_name += f"E{args.ang_excl_margin:.2f}"
            if args.ang_excl_loss_weight != 1.0:
                base_file_name += f"-{args.ang_excl_loss_weight:.2f}"
        if args.ang_incl_margin is not None:
            if 'rs' in str(args.ang_incl_margin):
                ang_incl_margin = str(args.ang_incl_margin).replace('rs','')
                base_file_name += f"Irs{float(ang_incl_margin):.2f}"
            elif 'r' in str(args.ang_incl_margin):
                ang_incl_margin = str(args.ang_incl_margin).replace('r','')
                base_file_name += f"Ir{float(ang_incl_margin):.2f}"
            
            elif 'ex' in str(args.ang_incl_margin):
                ang_incl_margin = str(args.ang_incl_margin).replace('ex','')
                base_file_name += f"Iex{float(ang_incl_margin):.2f}"
            elif 'e' in str(args.ang_incl_margin):
                ang_incl_margin = str(args.ang_incl_margin).replace('e','')
                base_file_name += f"Ie{float(ang_incl_margin):.2f}"
            else:
                base_file_name += f"I{float(args.ang_incl_margin):.2f}"
                
            if args.ang_incl_loss_weight != 1.0:
                base_file_name += f"-{args.ang_incl_loss_weight:.2f}"
        if args.ang_preserve_loss_weight is not None and args.ang_preserve_loss_weight > 0.0:
            base_file_name += f"P{args.ang_preserve_loss_weight:.2f}"
                
        base_file_name +='-'
        if args.ang_norm_loss_weight is not None:
            base_file_name += f"N{args.ang_norm_loss_weight:.2f}"
        if args.weight_modification_weight is not None:
            base_file_name += f"W{number_to_scientific_str(args.weight_modification_weight)}"
        if args.generic_loss_weight is not None:
            base_file_name += f"G{args.generic_loss_weight:.2f}"   
        if args.ang_loss_max_token_seq_len is not None and args.ang_loss_max_token_seq_len != 77 :
            base_file_name += f"nT{args.ang_loss_max_token_seq_len}"      


        
        if args.mask_out_template_prompt or args.mask_after_eos_nth_token is not None:
            base_file_name += f"-m" # mask out template prompt tokens for pushing loss
            
            if args.apply_push_only_eos:
                base_file_name += 'E'
                
            else:
                if args.mask_before_concept_token:
                    base_file_name += f"c" # mask out before concept token for pushing loss
                elif args.mask_out_template_prompt:
                    base_file_name += f"t" # mask out template prompt tokens for pushing loss
 
                if args.mask_after_eos_nth_token is not None:
                    base_file_name += f"e{args.mask_after_eos_nth_token}" # mask out after nth token after eos for pushing loss
            if args.rescale_from_masking:
                base_file_name += f".rs" # rescale from masking for pushing loss
        

            if args.pull_specific_token:

                if args.pull_specific_token == 'CE':
                    base_file_name += '.pCE'

                if args.pull_specific_token == 'Cs':
                    base_file_name += '.pCs'

                if args.pull_specific_token == 'C':
                    base_file_name += '.pC'             

                if args.pull_specific_token == 'E':
                    base_file_name += '.pE'

            elif args.mask_pull_before_concept_token:
                base_file_name += '.pc'


        if args.preserve_specific_token:
            base_file_name += f".a{args.preserve_specific_token}" # preserve specific token for pushing loss
        
        if args.use_refine_generic_concept:
            base_file_name += f"-rg" # use refine generic concept for pushing loss
        
        
        if args.ang_compute_token_option == 'avg_pairwise' or args.ang_compute_token_option == 'average_pairwise':
            base_file_name += '-At'
        # print(args.ang_push_pair_option)
        if args.ang_compute_token_option == 'allpair':
            base_file_name += f"-AT" # pairwise or allpair
            base_file_name += f".a{args.ang_aggr_token_option[0].capitalize()}" # pairwise or allpair
            
            
            
        # print(args.ang_push_pair_option)
        if args.ang_push_pair_option == 'allpair':
            base_file_name += f"-AC" # pairwise or allpair
            base_file_name += f".a{args.ang_all_pair_aggr_concept_option[0].capitalize()}" # pairwise or allpair
            
    
    if args.template_prompt is not None:
        # TE ... mean for push and pull (not for preserve) ... I already used some T for one that does not preserve with multiple prompts template
        if args.template_prompt == 'empty':
            base_file_name += '-TEe'
        elif args.template_prompt == 'basic':
            base_file_name += '-TEb'
        elif args.template_prompt == 'primitive':
            base_file_name += '-TEp'
    if args.weight_decay is not None or args.lr != 5e-5:
        base_file_name+= '_'
        if args.lr != 5e-5:
            base_file_name += f"lr{args.lr:.0e}"
        if args.weight_decay is not None:
            base_file_name += f"wd{args.weight_decay:.0e}"



    if args.apply_gradient_projection:
            base_file_name += f"_GP"
            base_file_name += '.g'
            if args.gradient_projection_param_group == 'base':
                base_file_name += f"B"
            if args.gradient_projection_param_group == 'global':
                base_file_name += f"G"
            if args.gradient_projection_param_group == 'attn_head':
                base_file_name += f"H"
            if args.gradient_projection_param_group == 'layer':
                base_file_name += f"L"
            if args.gradient_projection_param_group == 'neuron':
                base_file_name += f"N"
                
            base_file_name += '.p'
            if args.gradient_projection_mode == 'hard':
                base_file_name += f"H"
            if args.gradient_projection_mode == 'soft':
                base_file_name += f"S"
                
            if args.unlearn_proj_prob < 1.0:
                base_file_name += f"-u{args.unlearn_proj_prob:.2f}"
                
            base_file_name += '.'
            if  args.preservation_train_set:
                if args.preservation_train_set in ['00','01','02','03'] + ['G','UG','U']:
                    base_file_name += f"pe{args.preservation_train_set}"
                elif args.preservation_train_set == 'celeb':
                    base_file_name += f"cl"
                elif args.preservation_train_set == 'coco':
                    base_file_name += f"cc"
                    
                if args.preservation_split != 'train':
                    base_file_name += f".{args.preservation_split.upper()}"
                base_file_name += '-'

            if args.gradient_projection_preserve_scale is not None :
                
                if args.preservation_weight_option == 'convex' and args.gradient_projection_preserve_scale != 0.0 :
                    base_file_name += f"cPS{args.gradient_projection_preserve_scale:.2f}"
                else:
                    base_file_name += f"PS{args.gradient_projection_preserve_scale:.2f}"
                

                
            # if args.preservation_weight is not None and args.preservation_weight > 0:
            #     base_file_name += f"PS{args.preservation_weight:.2f}"



    if args.apply_poison:
        base_file_name += f".PNS"


        
    if args.decompositional_timestep_sampler is not None:
        base_file_name += f"_dT{args.decompositional_timestep_sampler}"


    if args.test_tag is not None:
        base_file_name += f"_{args.test_tag}"

        
    base_file_name += f"_U.{erase_concept_shortname}"
    
    
    base_file_name += "_sd1.4"  
    base_file_name += f".{args.train_precision}"
    
    # if args.lr != 5e-5:
    #     base_file_name += f".lr{args.lr:.0e}"

    if args.batch_size != 1:
        base_file_name += f".bs{args.batch_size}" # n just for note that still one preservation concept



      

      
    base_file_name += f"_r{args.seed}"
    


    
    
    
    
    return f"{base_file_name}"


def save_esd_model(esd_param_names, esd_params, args, training_step, total_grad_stats=None):
    model_name = resolve_model_name(args)

    if args.attacked_flag is not None:
         save_model_path = osp.join(args.save_path, model_name, 'attacked_models', f'{args.attacked_flag}_step{training_step}.safetensors')
    else:
        save_model_path = osp.join(args.save_path, model_name, f'step{training_step}.safetensors')
    os.makedirs(osp.dirname(save_model_path), exist_ok=True)

    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    save_file(esd_param_dict, save_model_path)
    print(f"ESD model saved at: {save_model_path}")
    
    
    if total_grad_stats is not None and len(total_grad_stats) > 0:
        grad_stats_path = osp.join(args.save_path, model_name, f'step{training_step}_grad_stats_{args.collect_gradient_statistics_option}.pt')
        os.makedirs(osp.dirname(grad_stats_path), exist_ok=True)
        torch.save(total_grad_stats, grad_stats_path)
        print(f"Gradient statistics saved at: {grad_stats_path}")
        
        # reset
        # total_grad_stats = defaultdict(list)


        


def load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16, device='cuda:0'):
    
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    base_unet.requires_grad_(False)
    
    esd_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    return pipe, base_unet, esd_unet

def get_esd_trainable_parameters(esd_unet, train_method='esd-x'):
    esd_params = []
    esd_param_names = []
    for name, module in esd_unet.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and 'attn2' in name:
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
            if train_method == 'esd-x-kv' and 'attn2' in name and ('to_k' in name or 'to_v' in name):
                for n, p in module.named_parameters():
                    # print(name)
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
                    
            if train_method == 'esd-s' and 'attn1' in name:
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
                          
                    
            if train_method == 'esd-xs' and ( 'attn2' in name or 'attn1' in name ):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
                    
            if train_method == 'esd-u' and ('attn2' not in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-all' :
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

    return esd_param_names, esd_params


def get_parser():
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for SDv1.4',
                    description = 'Finetuning stable-diffusion to erase the concepts')
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    # parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=50)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=3)
    
    parser.add_argument('--train_method', help='Type of method (esd-x, esd-u, esd-a, esd-x-strict)', type=str, required=True)
    parser.add_argument('--max_training_step', help='Number of max_training_step', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=5e-5)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=2)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/sd/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    

    parser.add_argument('--timestep_sampler', help='timestep constraint for diffusion model', type=str, required=False, default=None)
    parser.add_argument('--timestep_constraint', help='timestep constraint for diffusion model', type=str, required=False, default=None)
    parser.add_argument('--base_concept', type=str, choices=['null','general','erased','neighbor'], default='null', required=False)
    
    parser.add_argument('--erase_from', type=str, choices=[None,'uncond','object','general','forward','neighbor'], default=None, required=False)
    
    parser.add_argument('--preservation_weight', type=float,  default=None, required=False)
    parser.add_argument('--preservation_split', type=str,  default='train', choices=['train','test'] )
    
    parser.add_argument('--preservation_train_set', type=str,  default='U', choices=['celeb','coco'] + ['00','01','02','03']+['G','UG','U'] )
    parser.add_argument('--preservation_weight_option', type=str,  default='additive', choices=['additive','convex'])



    parser.add_argument('--decompositional_timestep_sampler',  type=str,  choices=[None,'avg','indiv'], default=None)

    parser.add_argument('--apply_gradient_projection',  action='store_true', default=False)
    parser.add_argument('--gradient_projection_mode', type=str, choices=['hard','soft','none'], default='hard')
    parser.add_argument('--gradient_projection_param_group', type=str, choices=['base','global','attn_head','layer','neuron'], default='attn_head')
    parser.add_argument('--gradient_projection_preserve_scale', type=float,  default=1.0)
    
    
    
    parser.add_argument('--seed', type=int,  default=123)
    parser.add_argument('--train_precision', type=str,  default='fp32', choices=['bf16','fp32'])
    parser.add_argument('--log_step', type=int,  default=100)
    parser.add_argument('--special_log_step', type=str,  default=None) # split by ','
    
    
    parser.add_argument('--unlearn_proj_prob', type=float,  default=1.00)

    parser.add_argument('--collect_gradient_statistics_option', type=str,  default=None, choices=[None, 'none','static', 'dynamic'])
    parser.add_argument('--test_tag', type=str,  default=None)
    parser.add_argument('--report_to', type=str,  default='wandb') # wandb
    parser.add_argument('--batch_size', type=int,  default=1) # wandb
    
    
    parser.add_argument('--extra_forward_prob', type=float, default=None) # wandb
    parser.add_argument('--use_indiv_extra_forward',action='store_true', default=False) # wandb
    parser.add_argument('--forward_general',action='store_true', default=False) # wandb
    parser.add_argument('--forward_preserve',action='store_true', default=False) # wandb
    parser.add_argument('--extra_forward_negative_guidance', type=float, default=None) # wandb
    
    
    parser.add_argument( "--load_unet_weight_path",type=str,default=None) # many unlearned model, UCE, ESD, 
    parser.add_argument( "--load_token_embedding_path",type=str,default=None) # many unlearned model, UCE, ESD, 
   


    parser.add_argument( "--apply_poison",action='store_true') # many unlearned model, UCE, ESD, 
    
    parser.add_argument( "--neg_guidance_method",type=str, default='cfg') #  'cfg'(default as ESD), ccfg
    
    parser.add_argument("--aei_loss_weight", type=float, default=0.0)
    parser.add_argument("--ang_excl_margin",type=float, default=0.0)
    parser.add_argument("--ang_incl_margin",type=str, default='0') 
    
    parser.add_argument("--ang_incl_loss_weight",type=float, default=1.00) 
    parser.add_argument("--ang_excl_loss_weight",type=float, default=1.00) 
    parser.add_argument("--ang_preserve_loss_weight",type=float, default=0.00) 
    
    parser.add_argument("--ang_push_pair_option", type=str, default='pairwise') # pairwise or allpair
    parser.add_argument("--ang_all_pair_aggr_concept_option", type=str, default='avg') # for all pair option, how to aggregate concept-level loss: mean or max
    
    
    parser.add_argument("--ang_norm_loss_weight",type=float, default=1.0)  # multiplied from aei_loss_weight
    parser.add_argument("--generic_loss_weight",type=float, default=1.0)  # other unlearn losses
    parser.add_argument("--weight_modification_weight",type=float, default=None)  # other unlearn losses
    

    parser.add_argument('--sim_param_group', type=str,  default='token')
    parser.add_argument('--ang_loss_max_token_seq_len', type=int, default=None) 
    
    
    
    parser.add_argument('--attacked_flag', type=int, default=None)  # flag for save model weight just for textual inversion attack
    parser.add_argument('--erase_tis', type=str, default=None) # we will use erase_ti for erase
    parser.add_argument('--erase_ti_weights', type=str, default=None) #  feature = W_iPe_i vs W_0Pe_i  (the weight that used to learn TIA)
    # parser.add_argument('--erased_unet_weights', type=str, default=None) #  feature = W_iPe_i vs W_0Pe_i  (the weight that used to learn TIA)
    
    # overwrite arugment
    parser.add_argument("--final_ang_preserve_loss_weight", type=float, default=None) 
    parser.add_argument("--final_ang_excl_margin",  type=float, default=None)
    

    
    parser.add_argument("--mask_out_template_prompt", action='store_true') # for pushing loss, whether to mask out the template prompt tokens or not (default as masking out)
    parser.add_argument("--mask_before_concept_token", action='store_true') # for pushing loss, whether to mask out the tokens before the concept token or not (default as masking out)
    parser.add_argument("--mask_pull_before_concept_token", action='store_true') # for pushing loss, whether to mask out the tokens before the concept token or not (default as masking out)
    parser.add_argument("--pull_specific_token",default=None,choices=['C','Cs','E','CE']) # for pushing loss, whether to mask out the tokens before the concept token or not (default as masking out)
    
    
    parser.add_argument("--mask_after_eos_nth_token",  type=int, default=None)
    parser.add_argument("--rescale_from_masking",  action='store_true')
    parser.add_argument("--apply_push_only_eos",  action='store_true')
    
    
    
    
    parser.add_argument("--ang_compute_token_option",  type=str, default='pairwise') # for computing angular loss, how to compute token-level loss: pairwise or allpair
    parser.add_argument("--ang_aggr_token_option",  type=str, default='avg', choices=['avg','max']) # for computing angular loss, if allpair, how to aggregate token-level loss: mean or max
    
    
    # ahM or ah
    parser.add_argument("--ang_aggr_feature_option",  type=str, default='avg', choices=['avg','max']) # for computing angular loss, if allpair, how to aggregate token-level loss: mean or max
    
    parser.add_argument("--use_refine_generic_concept",  action='store_true')
    
    
    parser.add_argument("--use_l2_loss",  action='store_true')

    parser.add_argument("--preserve_specific_token",default=None,choices=['B','T','TCE','C','E','CE']) # for pushing loss, whether to mask out the tokens before the concept token or not (default as masking out)
    
    parser.add_argument("--template_prompt",default=None,choices=[None,'empty','basic','primitive']) # for pushing loss, whether to mask out the tokens before the concept token or not (default as masking out)


    parser.add_argument("--weight_decay",type=float, default=None) 

    parser.add_argument("--only_save_base_sim",  action='store_true')



    
    return parser


#---


#+++ PPP_Loss(Ours) 
def compute_angular_exclusion_inclusion_loss(
    unet_u,         # unlearned UNet (trainable)
    unet_0,         # frozen reference UNet
    p_e,            # erased prompt embedding [B, T, 768]
    p_g,            # generic prompt embedding [B, T, 768]
    p_p,
    m_excl: float,
    m_incl: Union[float, str],
    # m_incl: float,
    layer_filter="attn2",  # cross-attention in diffusers SD1.4
    use_bias: bool = False,
    sim_param_group: str = "avg_token",  # {'avg_token', 'token', 'attn_head'}
    max_token_seq_len: int = None, 
    
    unet_u_tis=None,
    
    aggr_feature_option= 'avg',
    
    
    push_pair_option: str = 'pairwise', # pairwise or allpair
    n_prompt_template: int = None, # number of prompt template for each concept (including the raw text)
    all_pair_aggr_concept_option: str = 'avg', # for all pair option, how to aggregate concept-level loss: mean or max
    
    

    compute_token_option = 'pairwise',
    aggr_token_option = 'avg',
    
    apply_mask_before_concept_token=False,
    concept_token_idx=None, # dict of concept to token idxs for computing angular loss, if None, use all tokens
    
    
    
    template_prompt_end_idx=None, # for masking tokens belong to the prompt template for pushing loss not pushing the template prompt
    target_prompt_end_concentration_idx=None, 
    rescale_from_masking=False,
    
    
    generic_concept_token_idx=None, # dict of concept to token idxs for computing angular loss, if None, use all tokens
    generic_prompt_end_idx = None,
    apply_push_only_eos=False,
    return_l2_baseline = False,

    pull_specific_token=None,
    preserve_specific_token=None,

    relevant_generic_concept_idx=None,
    relevant_erased_concept_idx=None
    
    
):
    # all argument
    # print(f"apply_mask_before_concept_token: {apply_mask_before_concept_token}, concept_token_idx: {concept_token_idx}, template_prompt_end_idx: {template_prompt_end_idx}, target_prompt_end_concentration_idx: {target_prompt_end_concentration_idx}, rescale_from_masking: {rescale_from_masking}, generic_concept_token_idx: {generic_concept_token_idx}, apply_push_only_eos: {apply_push_only_eos}, return_l2_baseline: {return_l2_baseline}")
    # exit()
    """
    Angular exclusion + inclusion loss in KV projection space (unsquared hinge).

    sim_param_group:
      - 'avg_token' : mean over tokens -> cosine -> hinge
      - 'token'     : cosine per token -> hinge per token -> mean
      - 'attn_head' : mean over tokens -> reshape to (num_heads=8) -> cosine per head
                     -> hinge per head -> mean

    Also returns L_norm (NOT added to L_ang):
      L_norm = mean | log(||W_u p_e||) - log(||W_0 p_e||) |
    computed using the same grouping as sim_param_group.

    Returns:
      (L_excl, L_incl, L_ang, L_norm)
    """

    assert sim_param_group in {"avg_token", "token", "attn_head"}

    params_u = dict(unet_u.named_parameters())
    params_0 = dict(unet_0.named_parameters())
    
    
    if unet_u_tis is not None:
        # print(unet_u_tis[0].keys())
        params_u_tis = [unet_u_ti for unet_u_ti in unet_u_tis]
        
        # params_u_tis = [dict(unet_u_ti.named_parameters()) for unet_u_ti in unet_u_tis]

    excl_terms = []
    incl_terms = []
    norm_terms = []
    preserve_terms = []
    w_terms = []
    matched_layers = 0
    
    l_push_l2_terms = []
    l_pull_l2_terms = []
    l_preserve_l2_terms = []

    all_base_generic_target_similariy = [] # for analysis
    for name, W_u in params_u.items():
        if layer_filter not in name:
            continue
        if not (name.endswith("to_k.weight") or name.endswith("to_v.weight")):
            continue
        if name not in params_0:
            continue

        # Frozen reference
        W_0 = params_0[name].detach()
        
        
        

        
        
        # if unet_u_tis is not None:
            # W_u_tis = [params_u_ti[name].detach() for params_u_ti in params_u_tis]
            
            # W_u_tis = [params_u_ti.get(name, None) for params_u_ti in params_u_tis]
            # W_u_tis = [W_u_ti.detach() if W_u_ti is not None else None for W_u_ti in W_u_tis]

        # Bias handling
        if use_bias:
            b_name = name.replace(".weight", ".bias")
            b_u = params_u.get(b_name, None)
            b_0 = params_0.get(b_name, None)
            if b_0 is not None:
                b_0 = b_0.detach()
        else:
            b_u = None
            b_0 = None

        if unet_u_tis is not None:
            # remain the same
            W_u_e = F.linear(p_e, W_u, b_u)
            W_u_g = F.linear(p_g, W_u, b_u) 
            W_u_p = F.linear(p_p, W_u, b_u) 
            
            
            W_u_tis = [params_u_ti[name].detach().to(W_u.device) for params_u_ti in params_u_tis]
            
            with torch.no_grad():
                W_u_ti_e = [F.linear(p_e[:n_prompt_template], W_0, b_0) ] # first one is the raw text prompt and original pretrained
                for i_ti in range(len(W_u_tis)):
                    W_u_ti = W_u_tis[i_ti]
                    
                    # 1 is for the raw text, the rest is for tis
                    W_u_ti_e += [F.linear(p_e[(1+i_ti)*n_prompt_template: (1+i_ti+1)*n_prompt_template], W_u_ti, b_u)] # [B, T, D] for i_ti 
                    
                W_u_ti_e = torch.cat(W_u_ti_e, dim=0) 
            
            # print(W_u_e.shape) # [B, T, D]
            # print('applying erased weight projection')
            
        else:
        
            # Linear projections: [B, T, D]
            W_u_e = F.linear(p_e, W_u, b_u)
            W_u_g = F.linear(p_g, W_u, b_u) 
            W_u_p = F.linear(p_p, W_u, b_u) # for preservation term
            
            
            
        # from pretrained 
        with torch.no_grad():
            W_0_e = F.linear(p_e, W_0, b_0)
            W_0_g = F.linear(p_g, W_0, b_0)
            W_0_p = F.linear(p_p, W_0, b_0)
            

        if sim_param_group == "avg_token":
            W_u_e_m = W_u_e.mean(dim=1)
            W_0_e_m = W_0_e.mean(dim=1)
            W_0_g_m = W_0_g.mean(dim=1)

            cos_excl = F.cosine_similarity(W_u_e_m, W_0_e_m, dim=-1)
            cos_incl = F.cosine_similarity(W_u_e_m, W_0_g_m, dim=-1)

            excl_terms.append(torch.clamp(cos_excl - m_excl, min=0.0).mean())
            incl_terms.append(torch.clamp(m_incl - cos_incl, min=0.0).mean())

            norm_u = W_u_e_m.norm(dim=-1)
            norm_0 = W_0_e_m.norm(dim=-1)
            norm_terms.append(torch.abs(torch.log(norm_u) - torch.log(norm_0)).mean())

        elif sim_param_group == "token":
            
            
            B, T, D = W_u_e.shape
            
            n_concepts = p_e.shape[0] // n_prompt_template
            
            
            
            # Exclusion (Push)
            # by default
            if push_pair_option == 'pairwise' or n_concepts == 1:
                
                if compute_token_option  == 'pairwise' and return_l2_baseline :
                  # Calculate squared L2 distance: ||W_e*p_e - W_0*p_e||^2
                    m_push = m_excl 
                    m_pull = m_incl if isinstance(m_incl, (int, float)) else float(m_incl[1:]) # better use 0.0
                    
                    # prevent stagenation of the first iteration                        
                    if m_pull == 0.0:
                        with torch.no_grad():
                            if torch.allclose(W_u_e, W_0_e):
                                W_u_e.add_(1e-4 * torch.randn_like(W_u_e))
                    # =========================
                    # L2-push
                    # =========================
                    # ||W^j_{e,l} p_e - W^j_{0,l} p_e||_2^2
                    dist_sq_push = torch.sum((W_u_e - W_0_e) ** 2, dim=-1)

                    # max(m_push - dist_sq_push, 0)
                    l_push_l2 = torch.clamp(m_push - dist_sq_push, min=0.0)

                    # Average over batch/examples, keep one scalar for this (l, j)
                    l_push_l2_terms.append(l_push_l2.mean())

                    # =========================
                    # L2-pull
                    # =========================
                    # ||W^j_{e,l} p_e - W^j_{0,l} p_g||_2^2
                    dist_sq_pull = torch.sum((W_u_e - W_0_g) ** 2, dim=-1)

                    # max(dist_sq_pull - m_pull, 0)
                    l_pull_l2 = torch.clamp(dist_sq_pull - m_pull, min=0.0)

                    # Average over batch/examples, keep one scalar for this (l, j)
                    l_pull_l2_terms.append(l_pull_l2.mean())
            
                
                # print(f"concept_token_idx:  {concept_token_idx}") # [2, 5, 6, 5, 5]
                # print(f"eos_idx:  {eos_token_idx}") # [3, 6, 7, 6, 6]
                # print(f"W_u_e.shape: {W_u_e.shape}") # [5, 77, 1280]

                if compute_token_option in ['avg_pairwise', 'average_pairwise']:
                    n_prompt = W_u_e.shape[0]
                    device = W_u_e.device

                    # ---------------------------
                    # Helper: extract concept tokens
                    # ---------------------------
                    def extract_avg_tokens(W, token_idx):
                        """
                        W: [n_prompt, n_tokens, d]
                        token_idx:
                            - list[int], one token per prompt
                            - list[list[int]], multiple concept tokens per prompt

                        returns: [n_prompt, d]
                        """
                        out = []

                        for i, idx in enumerate(token_idx):
                            if isinstance(idx, (list, tuple)):
                                idx = torch.tensor(idx, device=device)
                                emb = W[i, idx].mean(dim=0)     # average tokens within prompt
                                print("average tokens within prompt")

                            else:
                                emb = W[i, int(idx)]            # single token

                            out.append(emb)

                        return torch.stack(out, dim=0)          # [n_prompt, d]

                    # hack: concept_token_idx
                    concept_token_idx_1 = [i_ -1 for i_ in concept_token_idx ]
                    concept_token_idx_ = [(a,b) for a,b in zip(concept_token_idx_1,concept_token_idx)]
                    print(concept_token_idx_)
                    # ---------------------------
                    # 1. Concept embedding
                    # ---------------------------
                    concept_emb_u = extract_avg_tokens(W_u_e, concept_token_idx_)   # [n_prompt, d]
                    concept_emb_0 = extract_avg_tokens(W_0_e, concept_token_idx_)   # [n_prompt, d]

                    # ---------------------------
                    # 2. EOS embedding
                    # ---------------------------
                    eos_token_idx = []

                    for idx in concept_token_idx:
                        if isinstance(idx, (list, tuple)):
                            eos_token_idx.append(max(idx) + 1)
                        else:
                            eos_token_idx.append(int(idx) + 1)

                    eos_token_idx = torch.tensor(eos_token_idx, device=device)
                    batch_idx = torch.arange(n_prompt, device=device)

                    eos_emb_u = W_u_e[batch_idx, eos_token_idx]   # [n_prompt, d]
                    eos_emb_0 = W_0_e[batch_idx, eos_token_idx]   # [n_prompt, d]

                    # ---------------------------
                    # 3. Average across templates
                    # ---------------------------
                    avg_concept_u = concept_emb_u.mean(dim=0)     # [d]
                    avg_concept_0 = concept_emb_0.mean(dim=0)     # [d]

                    avg_eos_u = eos_emb_u.mean(dim=0)             # [d]
                    avg_eos_0 = eos_emb_0.mean(dim=0)             # [d]

                    # ---------------------------
                    # 4. Build representation
                    # ---------------------------
                    vec_u = []
                    vec_0 = []

                    w_c = 1.0
                    w_e = 1.0

                    selected_push = ['C']

                    if 'C' in selected_push:
                        vec_u.append(w_c * avg_concept_u)
                        vec_0.append(w_c * avg_concept_0)

                    if 'E' in selected_push:
                        vec_u.append(w_e * avg_eos_u)
                        vec_0.append(w_e * avg_eos_0)

                    vec_u = torch.stack(vec_u, dim=0)   # [n_selected, d]
                    vec_0 = torch.stack(vec_0, dim=0)   # [n_selected, d]

                    # ---------------------------
                    # 5. Cosine similarity
                    # ---------------------------
                    cos_excl_t = F.cosine_similarity(vec_u, vec_0, dim=-1)  # [n_selected]

                    excl_term = torch.clamp(cos_excl_t - m_excl, min=0.0)
                    excl_terms.append(excl_term.mean())

                    print(f"vec_u.shape: {vec_u.shape}")
                    print(f"pushing loss: {excl_term}")

                    # just hacky
                    P, T = W_u_e.shape[:2]

                elif compute_token_option  == 'pairwise':
                
                    cos_excl_t = F.cosine_similarity(W_u_e, W_0_e, dim=-1)
                    excl_term = torch.clamp(cos_excl_t - m_excl, min=0.0) #  B, T, 1


                    P, T = excl_term.shape[:2]
                    invalid = torch.zeros((P, T), dtype=torch.bool, device=excl_term.device)





                    if apply_mask_before_concept_token and concept_token_idx is not None:
                        concept_token = torch.as_tensor(concept_token_idx, device=excl_term.device).unsqueeze(1) 
                        mask_before_concept_token = torch.arange(T, device=excl_term.device).unsqueeze(0) < concept_token           # (P,T)
                        
                        # print(concept_token)
                        # print(mask_before_concept_token)
                        invalid |= mask_before_concept_token
                        
                        excl_term = excl_term.masked_fill(mask_before_concept_token, 0.0)

                    if template_prompt_end_idx is not None:
                        # print(f'mask out the template prompt during pushing operation: {template_prompt_end_idx}')
                        end = torch.as_tensor(template_prompt_end_idx, device=excl_term.device).unsqueeze(1)  # (P,1)
                        mask_template = torch.arange(T, device=excl_term.device).unsqueeze(0) < end           # (P,T)
                        invalid |= mask_template

                        excl_term = excl_term.masked_fill(mask_template, 0.0)
                        # print(f'after masked:\n{excl_term}')

                    if target_prompt_end_concentration_idx is not None or apply_push_only_eos :
                        # print(f'mask out the tokens after certain position for concentrating the pushing loss on the early tokens: {target_prompt_end_concentration_idx}')
                        end = torch.as_tensor(target_prompt_end_concentration_idx, device=excl_term.device).unsqueeze(1)  # (P,1)
                        mask_latetoken = torch.arange(T, device=excl_term.device).unsqueeze(0) > end                       # (P,T)
                        invalid |= mask_latetoken

                        excl_term = excl_term.masked_fill(mask_latetoken, 0.0)
                        
                        if apply_push_only_eos:
                            print('using only eos')
                            start = torch.as_tensor(target_prompt_end_concentration_idx, device=excl_term.device).unsqueeze(1)  # (P,1)
                            mask_earlytoken = torch.arange(T, device=excl_term.device).unsqueeze(0) < start                       # (P,T)
                            
                            invalid |= mask_earlytoken
                            excl_term = excl_term.masked_fill(mask_earlytoken, 0.0)
            

                    if aggr_feature_option == 'max':  # max over tokens (your original)
                        excl_term = excl_term.max(dim=1).values  # NOTE: dim=1 is tokens if excl_term is (P,T,...)
                    else:


                        # print(invalid.shape)
                        # print(f"push: {invalid}") # 55, 77
                        print(invalid)
                        # print(apply_push_only_eos) # False
                        
                        print(f"L_push: {excl_term[~invalid]}") # print only
                        print(excl_term)

                        if rescale_from_masking:
                            # ---- key fix: correct mean to ignore masked tokens ----
                            num_valid = (~invalid).sum(dim=1).clamp(min=1)     # (P,)
                            scale = (T / num_valid)                            # (P,)
                            # if excl_term has extra dims (e.g., (P,T,D)), broadcast scale
                            while scale.ndim < excl_term.ndim - 1:
                                scale = scale.unsqueeze(-1)

                            excl_term = excl_term.mean(dim=1) * scale          # corrected mean over tokens

                    # (max) mean over prompts (same as you had)
                    excl_terms.append(excl_term.mean())


                
                # overtoken
                elif compute_token_option  == 'allpair':
                    x1 = W_u_e.view(n_prompt_template, T, D)  #  ( P, T, D) # 1 concepts
                    x1 = F.normalize(x1, dim=-1)  
                    
                    
                    
                    x2 = W_0_e.view(n_prompt_template, T, D)  #  ( P, T, D)
                    x2 = F.normalize(x2, dim=-1) 
                    
                    cos_excl_t = x1 @ x2.transpose(-1, -2)   # (P, T, T)
    
    
    
                    # masking anything involving template tokens
                    if template_prompt_end_idx is not None:
                        P, T, _ = cos_excl_t.shape

                        end = torch.as_tensor(template_prompt_end_idx, device=cos_excl_t.device, dtype=torch.long).clamp_(0, T)  # (P,)
                        m = torch.arange(T, device=cos_excl_t.device).unsqueeze(0) < end.unsqueeze(1)  # (P, T)

                        full_mask = m.unsqueeze(-1) | m.unsqueeze(-2)   # (P, T, T)  rows OR cols are template tokens
                        cos_excl_t = cos_excl_t.masked_fill(full_mask, 0.0)
            
                    
                    if aggr_token_option == 'avg':
                        excl_term = torch.clamp(cos_excl_t - m_excl, min=0.0).mean(dim=-1) # mean over the T tokens in reference feature for each token in target feature, resulting in (P, T)
                    elif aggr_token_option == 'max':
                        
                        excl_term =torch.clamp(cos_excl_t - m_excl, min=0.0).max(dim=-1).values #  max over the T tokens
                        # (P,T)
                        
                    # mask out template token
                    if template_prompt_end_idx is not None:
                        print(f'mask out the template prompt during pushing operation: {template_prompt_end_idx}') # [1, 4, 5, 4, 4] (P)
                        end = torch.tensor(template_prompt_end_idx).unsqueeze(1).to(excl_term.device)
                        mask = torch.arange(excl_term.size(1), device=excl_term.device) < end
                        excl_term = excl_term.masked_fill(mask, 0.0)
                        
                        # the entire row that related to the template prompt tokens will be masked out in the cos_excl_t, which means those template prompt tokens will not contribute to the pushing loss for all tokens in the target feature
                        # excl_term = excl_term.masked_fill(mask.unsqueeze(1), 0.0)
                        
                        # the entire column that related to the template prompt tokens will be masked out in the cos_excl_t, which means those template prompt tokens will not contribute to the pushing loss for all tokens in the target feature
                        excl_term = excl_term.masked_fill(mask.unsqueeze(0), 0.0)

                        # print(excl_term)

                #hack: should make argument for this
                
                if aggr_feature_option == 'max': # if avg .... just do it at the end
                    excl_term = excl_term.max(dim=-1).values  # max over tokens
                    
                    
                excl_terms.append(excl_term.mean()) # mean over prompts template
                
                # excl_terms.append(torch.clamp(cos_excl_t - m_excl, min=0.0).mean())

            # aggregation all multiple concepts 
            elif push_pair_option == 'allpair' and n_concepts > 1:
                # print('push all option')
                
            # target features (computed with the learnable weights)
                x1 = W_u_e.view(n_concepts, n_prompt_template, T, D)  #  (C, P, T, D)
                x1 = x1.permute(1, 2, 0, 3)            # # (P, T, C, D)
                x1 = F.normalize(x1, dim=-1)      
                
                x2 = W_0_e.view(n_concepts, n_prompt_template, T, D)  #  (C, P, T, D)
                x2 = x2.permute(1, 2, 0, 3)            # # (P, T, C, D)
                x2 = F.normalize(x2, dim=-1)                
                
                
                cos_cc_same_template = x1 @ x2.transpose(-1, -2) #   # (P, T, C, C)
                # print(cos_cc_same_template)
                
                
                ## concept aggregation
                if all_pair_aggr_concept_option == 'avg':
                    excl_terms.append(torch.clamp(cos_cc_same_template - m_excl, min=0.0).mean())
                
                elif all_pair_aggr_concept_option == 'max':
                    
                    # print(torch.clamp(cos_cc_same_template, min=0.0).max(dim=-1).values)
                    # For each concept in target feature(x1), find the most similar reference feature (x2)
                    excl_terms.append(torch.clamp(cos_cc_same_template - m_excl, min=0.0).max(dim=-1).values.mean())

                    # print(excl_terms[-1])


            
            


            # Inclusion (PULL)
            
            
            # possible shifted (make sure the target concept idx == generic concept idx )
            # if generic_concept_token_idx not None:
                
                # W_0_e # [B, T, D]
            
            
            invalid = torch.zeros((P, T), dtype=torch.bool, device=excl_term.device)


            if pull_specific_token == 'Cs':

                cos_incl_t = []
                base_cos_incl_t = []

                for e_prompt_idx, erased_idx in enumerate(relevant_erased_concept_idx):
                    erased_u = W_u_e[e_prompt_idx, erased_idx]   # [n_e, d]
                    erased_0 = W_0_e[e_prompt_idx, erased_idx]   # [n_e, d]

                    for g_prompt_idx, generic_idx in enumerate(relevant_generic_concept_idx):
                        generic_0 = W_0_g[g_prompt_idx, generic_idx]  # [n_g, d]

                        cos_incl_t.append(
                            F.cosine_similarity(
                                erased_u[:, None, :],
                                generic_0[None, :, :],
                                dim=-1,
                            )
                        )

                        with torch.no_grad():
                            base_cos_incl_t.append(
                                F.cosine_similarity(
                                    erased_0[:, None, :],
                                    generic_0[None, :, :],
                                    dim=-1,
                                )
                            )

                cos_incl_t = torch.stack(cos_incl_t)
                base_cos_incl_t = torch.stack(base_cos_incl_t)
                # print(f'cos_incl_t: {cos_incl_t}')
                # print(f'base_cos_incl_t: {base_cos_incl_t}')


    
            
            elif generic_concept_token_idx is None:
            
                cos_incl_t = F.cosine_similarity(W_u_e, W_0_g, dim=-1)
                
                # for 'r' and 'rs'
                with torch.no_grad():
                    base_cos_incl_t = F.cosine_similarity(W_0_e, W_0_g, dim=-1)
                
            else:
                
                B, T, D = W_u_e.shape

                shifts = torch.tensor(concept_token_idx).view(B) -  torch.tensor(generic_concept_token_idx).view(B)  # [B]
                shifts = shifts.to(W_0_g.device)
                
                # print(f'generic to target shifts: {shifts}')
                

                # Build shifted W_0_g via extended buffer
                T_ext = T + shifts.abs().max().item()
                W_0_g_ext = torch.zeros(B, T_ext, D, device=W_0_g.device, dtype=W_0_g.dtype) + 1e-12 # plus 1e-12 to prevent 0/0 -> nan

                for b in range(B):
                    s = shifts[b].item()
                    src_start = max(0, -s)
                    dst_start = max(0,  s)
                    length = T - max(0, abs(s))
                    if length > 0:
                        W_0_g_ext[b, dst_start:dst_start+length, :] = W_0_g[b, src_start:src_start+length, :]

                W_0_g_shifted = W_0_g_ext[:, :T, :]  # [B, T, D]

                # Mark shifted-out positions directly into existing `invalid`
                positions = torch.arange(T, device=W_0_g.device).unsqueeze(0)   # [1, T]
                src_positions = positions - shifts.unsqueeze(1)                   # [B, T]
                
                
                
                cos_incl_t = F.cosine_similarity(W_u_e, W_0_g_shifted, dim=-1)  # [B, T]
                
                # for 'r' and 'rs'
                with torch.no_grad():
                    base_cos_incl_t = F.cosine_similarity(W_0_e, W_0_g_shifted, dim=-1)
                    print(f"base cossim: {base_cos_incl_t}")
                    # print(base_cos_incl_t)
                
                
                # mask that will used later
                generic_shift = (src_positions < 0) | (src_positions >= T)  
                invalid |= generic_shift          # [B, T]
                
                # print(generic_shift)
                
                
                
                # mask out tokens before concept token (applies to both branches)
                concept_token = torch.as_tensor(concept_token_idx, device=cos_incl_t.device).view(P, 1)
                mask_before_concept_token = torch.arange(T, device=cos_incl_t.device).unsqueeze(0) < concept_token  # [P, T]
                invalid |= mask_before_concept_token
                
                if pull_specific_token is not None:
                    if pull_specific_token == 'C':
                        # print('pull C')
                        mask_except_concept = torch.arange(T, device=cos_incl_t.device).unsqueeze(0) != concept_token  # [P, T]
                        invalid |= mask_except_concept
                    elif pull_specific_token == 'E':
                        # print('pull E')
                        mask_except_eos = torch.arange(T, device=cos_incl_t.device).unsqueeze(0) != concept_token + 1  # [P, T]
                        invalid |= mask_except_eos

                    elif pull_specific_token == 'CE':
                        # print('pull CE')
                        mask_except_concept_and_eos = (
                            (torch.arange(T, device=cos_incl_t.device).unsqueeze(0) != concept_token) &
                            (torch.arange(T, device=cos_incl_t.device).unsqueeze(0) != concept_token + 1)
                        )  # [P, T]
                        invalid |= mask_except_concept_and_eos


                    # elif pull_specific_token == 'Cs': 
                    #     print('apply Cs')

                    #     # generic_concept_token_idx
                    #     generic_prompt_eos_idx=  torch.as_tensor(generic_prompt_end_idx, device=cos_excl_t.device, dtype=torch.long)
                    #     target_prompt_eos_idx =  torch.as_tensor(concept_token_idx, device=cos_excl_t.device, dtype=torch.long)+1 # eos = last_concept_token + 1

                    #     template_prompt_end_idx_ = torch.as_tensor(template_prompt_end_idx, device=cos_excl_t.device, dtype=torch.long)

                    #     # expect share template
                    #     generic_concept_tokens =  torch.arange(T, device=cos_incl_t.device).unsqueeze(0) 
                    #     generic_concept_tokens = (generic_concept_tokens > template_prompt_end_idx_) | (generic_concept_tokens < generic_prompt_eos_idx)
                       
                    #     target_concept_tokens =  torch.arange(T, device=cos_incl_t.device).unsqueeze(0) 
                    #     target_concept_tokens = (target_concept_tokens > template_prompt_end_idx_) | (target_concept_tokens < target_prompt_eos_idx)
                       

                    #     print(f"generic_concept_tokens:\n{generic_concept_tokens}")
                    #     print(f"target_concept_tokens:\n{target_concept_tokens}")


                    # print(invalid)
                            
            if 'rs' in m_incl:
                # print('using improved symmetric relative generic inclusion loss')
                
                m_incl_ =  float(m_incl.replace('rs',''))
                

                # r1 = torch.clamp(m_incl_ - cos_incl_h/base_cos_incl_h, min=0.0).mean()
                r1 = torch.clamp(m_incl_ - cos_incl_t/base_cos_incl_t, min=0.0)
                r2 = torch.clamp(cos_incl_t/base_cos_incl_t - (2-m_incl_), min=0.0)
                
                # either r1 or r2 is greater than 0 will contribute to the loss, which means either cos_incl_h/base_cos_incl_h is smaller than m_incl_ or greater than (2-m_incl_)
                # incl_terms.append((r1 + r2).mean())


                if pull_specific_token == 'Cs': # every element is selected already
                    print(f"L_pull: {r1+r2}")
                    incl_terms.append((r1+r2).mean())
                else:
                    print(f"symmetric L_pull: {(r1 + r2)[~invalid]}")
                    incl_terms.append((r1 + r2)[~invalid].mean())


                
            # releative: Lpull = max (m_pull - cos_sim (WePe, W0Pg) / cos_sim (W0Pe, W0Pg) , 0)
            elif 'r' in m_incl:
                # print('using improved relative generic inclusion loss')
                m_incl_ =  float(m_incl.replace('r',''))
                # cos_incl_h = F.cosine_similarity(W_u_e_h, W_0_g_h, dim=-1)


                r = torch.clamp(m_incl_ - cos_incl_t/base_cos_incl_t, min=0.0)
                # incl_terms.append(r.mean())
                
                # print(r)
                # print(f"valid tokens: {(~invalid).sum().item()} / {invalid.numel()}")


                if pull_specific_token == 'Cs': # every element is selected already
                    print(f"L_pull: {r}")
                    incl_terms.append(r.mean())

                    all_base_generic_target_similariy += [base_cos_incl_t.detach()]



                else:

                    print(f"L_pull: {r[~invalid]}")

                    incl_terms.append(r[~invalid].mean())


                    all_base_generic_target_similariy += [base_cos_incl_t[~invalid].detach()]
                
                
       
       
            else:
                # incl_terms.append(torch.clamp(float(m_incl) - cos_incl_t, min=0.0).mean())
                a =  torch.clamp(float(m_incl) - cos_incl_t, min=0.0)
                incl_terms.append(a[~invalid].mean())
                
                print(f"Absolute L_pull: {a[~invalid]}")
                
            # masking
            
            # if generic_concept_token_idx not None:
                
                
                
                
                
                
                
                         
            # excl_terms.append(torch.clamp(cos_excl_tok - m_excl, min=0.0).mean())
            # incl_terms.append(torch.clamp(m_incl - cos_incl_tok, min=0.0).mean())

            norm_u = W_u_e.norm(dim=-1)
            norm_0 = W_0_e.norm(dim=-1)
            norm_terms.append(torch.abs(torch.log(norm_u) - torch.log(norm_0)).mean())
            
       
            N_Pp, T = W_u_p.shape[:2]
            preserve_invalid = torch.zeros((N_Pp, T), dtype=torch.bool, device=excl_term.device)

            if preserve_specific_token is not None:

                print('THIS HAVE TO ADAPT TO P_P FIRST')
                if preserve_specific_token == 'B':
                    preserve_invalid = torch.ones((P, T), dtype=torch.bool, device=excl_term.device)
                    for i in range(P):
                        preserve_invalid[i][0] = False

                # mask out after eos token
                if preserve_specific_token == 'TCE':

                    concept_token_idx_ = torch.as_tensor(concept_token_idx, device=cos_incl_t.device).view(P, 1)
                    eos_token_idx =  concept_token_idx_ + 1 

                    mask_after_eos = torch.arange(T, device=cos_incl_t.device).unsqueeze(0) > eos_token_idx  # [P, T]
                    preserve_invalid |= mask_after_eos

                    # print(f"preserve_invalid: {preserve_invalid}")

            # preservation term
            cos_preserve = F.cosine_similarity(W_u_p, W_0_p, dim=-1)
            preserve_term =  1 - cos_preserve
            preserve_terms.append(preserve_term[~preserve_invalid].mean())

            print(f"L_preserve: {preserve_term[~preserve_invalid]}")

                   
            # preservation term
            # cos_preserve = F.cosine_similarity(W_u_g, W_0_g, dim=-1)
            # preserve_term =  1 - cos_preserve
            # preserve_terms.append(preserve_term.mean())
                
            

            preserve_l2_term = torch.sum((W_u_p - W_0_p) ** 2, dim=-1)
            l_preserve_l2_terms.append(preserve_l2_term.mean())
                    #   l_push_l2 = torch.clamp(m_push - dist_sq, min=0.0)
                    #     l_push_l2_terms += [l_push_l2.mean()]
                        
                    #     # Lpull
                    #     dist_sq_pull = torch.sum((W_e_p_e - W_0_p_g) ** 2, dim=-1)
                    #     l_pull_l2 = torch.clamp(dist_sq_pull - m_pull, min=0.0)
                    #     l_pull_l2_terms += [l_pull_l2.mean()]
                        
                     
            
            

        elif sim_param_group == "attn_head":
            # Keep token resolution; reshape into heads per token.
            # W_*: [B, T, D] -> [B, T, H, d]
            num_heads = 8
            B, T, D = W_u_e.shape
            assert D % num_heads == 0
            head_dim = D // num_heads

            W_u_e_h = W_u_e.view(B, T, num_heads, head_dim)
            W_0_e_h = W_0_e.view(B, T, num_heads, head_dim)
            W_0_g_h = W_0_g.view(B, T, num_heads, head_dim)
            W_u_g_h = W_u_g.view(B, T, num_heads, head_dim)
            # print(W_u_e_h.shape) # [5, 77, 8, 160]
            
            # # Cosine per (token, head): [B, T, H]
            # if max_token_seq_len is not None:
            #     print(f'using only {max_token_seq_len} tokens for angular loss computation')
            #     cos_excl_h = F.cosine_similarity(W_u_e_h[:, :max_token_seq_len], W_0_e_h[:, :max_token_seq_len], dim=-1)
            #     cos_incl_h = F.cosine_similarity(W_u_e_h[:, :max_token_seq_len], W_0_g_h[:, :max_token_seq_len], dim=-1)
            # else:
            #     cos_excl_h = F.cosine_similarity(W_u_e_h, W_0_e_h, dim=-1)
            #     cos_incl_h = F.cosine_similarity(W_u_e_h, W_0_g_h, dim=-1)
        
                    # by default, we use pairwise push, which means each target concept is only pushed away from its own original feature and pulled towards its own generic feature, which is more stable and effective especially when the number of concepts increases
            if push_pair_option == 'pairwise' or n_concepts == 1:
                cos_excl_h = F.cosine_similarity(W_u_e_h, W_0_e_h, dim=-1) #  [B, T, nH, 1] 
                cos_incl_h = F.cosine_similarity(W_u_e_h, W_0_g_h, dim=-1)
            
                # head-wise generic inclusion margin according to the original similarity between the target concept and the generic concept
                excl_term = torch.clamp(cos_excl_h - m_excl, min=0.0)
              
              
                P, T = excl_term.shape[:2]
                invalid = torch.zeros((P, T), dtype=torch.bool, device=excl_term.device)
                # print(invalid.shape, excl_term.shape) # torch.Size([5, 77]) torch.Size([5, 77, 8])
                # template_prompt_end_idx want to have size of [B,77] (5,77)
                if template_prompt_end_idx is not None:
                    print(f'mask out the template prompt during pushing operation: {template_prompt_end_idx}') # [1, 4, 5, 4, 4] (P)
                    
                    end = torch.as_tensor(template_prompt_end_idx, device=excl_term.device)  # (B,)
                    mask_template = torch.arange(excl_term.size(1), device=excl_term.device)[None, :] < end[:, None]  # (B,77)

                    excl_term = excl_term.masked_fill(mask_template.unsqueeze(-1), 0.0)  # (B,77,1) -> broadcast to (B,77,D)
                    print(mask_template.shape) # [5, 77]
                    invalid |= mask_template
                  
                    
                    # the entire row that related to the template prompt tokens will be masked out in the cos_excl_t, which means those template prompt tokens will not contribute to the pushing loss for all tokens in the target feature
                    
                    # the entire column that related to the template prompt tokens will be masked out in the cos_excl_t, which means those template prompt tokens will not contribute to the pushing loss for all tokens in the target feature
                    # # excl_term = excl_term.masked_fill(mask.unsqueeze(0), 0.0)
                    # print(excl_term.shape)
                    # print(excl_term)

                if target_prompt_end_concentration_idx is not None:
                    # print(f'mask out the tokens after certain position for concentrating the pushing loss on the early tokens: {target_prompt_end_concentration_idx}')
                    end = torch.as_tensor(target_prompt_end_concentration_idx, device=excl_term.device)  # (P,1)
                    mask_latetoken = torch.arange(excl_term.size(1), device=excl_term.device)[None, :] > end[:, None]                   # (P,T)
                    invalid |= mask_latetoken

                    excl_term = excl_term.masked_fill(mask_latetoken.unsqueeze(-1), 0.0)
                    print(f'after eos masked:\n{excl_term}')

                if rescale_from_masking:
                    # ---- key fix: correct mean to ignore masked tokens ----
                    num_valid = (~invalid).sum(dim=1).clamp(min=1)     # (P,)
                    scale = (T / num_valid)                            # (P,)
                    # if excl_term has extra dims (e.g., (P,T,D)), broadcast scale
                    while scale.ndim < excl_term.ndim - 1:
                        scale = scale.unsqueeze(-1)

                    excl_term = excl_term.mean(dim=1) * scale          # corrected mean over tokens



                excl_terms += [excl_term.mean()] # mean over tokens and heads    

                
            
            
            # exclusion (PULL)
            n_concepts = p_e.shape[0] // n_prompt_template
            if push_pair_option == 'allpair' and n_concepts > 1:
                # remain the same
                cos_incl_h = F.cosine_similarity(W_u_e_h, W_0_g_h, dim=-1)

                
                # from  (B, num_tokens, num_heads, head_dim) -> (n_concepts, n_prompt_template, num_tokens, num_heads, head_dim)
                # B form: [ P1C1, P2C1, P3C1, P1C2, P2C2, .. ]
                
                # target features (computed with the learnable weights)
                x1 = W_u_e_h.view(n_concepts, n_prompt_template, T, num_heads, head_dim) # (C, P, T, H, D) # [1, 5, 77, 8, 40]' is invalid for input of size 246400
                x1 = x1.permute(1, 2, 3, 0, 4)          # (P, T, H, C, D)
                x1 = F.normalize(x1, dim=-1)                  # normalize on D (head_dim)
                
                
                if unet_u_tis is not None:
                    # reference features (computed with frozen weights)
                    W_u_ti_e_h = W_u_ti_e.view(B, T, num_heads, head_dim)
                    x2 = W_u_ti_e_h.view(n_concepts, n_prompt_template, T, num_heads, head_dim)
                    x2 = x2.permute(1, 2, 3, 0, 4)          # (P, T, H, C, D)
                    x2= F.normalize(x2, dim=-1)            

                    cos_cc_same_template = x1 @ x2.transpose(-1, -2)   # (P, T, H, C, C)
                
                else:
                      cos_cc_same_template = x1 @ x1.transpose(-1, -2)
                
                
                ## concept aggregation
                if all_pair_aggr_concept_option == 'avg':
                    excl_terms.append(torch.clamp(cos_cc_same_template - m_excl, min=0.0).mean())
                
                elif all_pair_aggr_concept_option == 'max':
                    
                    # For each concept in target feature(x1), find the most similar reference feature (x2)
                    excl_terms.append(torch.clamp(cos_cc_same_template - m_excl, min=0.0).max(dim=-1).values.mean())

                    # excl_terms.append(torch.clamp(cos_cc_same_template - m_excl, min=0.0).max(dim=-1).mean())

                # cos_excl_h = F.cosine_similarity(W_u_e_h, W_0_e_h, dim=-1) # (n_concepts, n_prompt_template, num_tokens, num_heads)
                # cos_excl_h = F.cosine_similarity(W_u_e_h, W_0_e_h, dim=-1)
            



              
            # Inclusion (Pull)  
              
            # symmetric 
            if 'rs' in m_incl:
                # print('using improved symmetric relative generic inclusion loss')
                
                m_incl_ =  float(m_incl.replace('rs',''))
                
                with torch.no_grad():
                    base_cos_incl_h = F.cosine_similarity(W_0_e_h, W_0_g_h, dim=-1)
                # r1 = torch.clamp(m_incl_ - cos_incl_h/base_cos_incl_h, min=0.0).mean()
                r1 = torch.clamp(m_incl_ - cos_incl_h/base_cos_incl_h, min=0.0)
                r2 = torch.clamp(cos_incl_h/base_cos_incl_h - (2-m_incl_), min=0.0)
                
                # either r1 or r2 is greater than 0 will contribute to the loss, which means either cos_incl_h/base_cos_incl_h is smaller than m_incl_ or greater than (2-m_incl_)
                incl_terms.append((r1 + r2).mean())

                
            # releative: Lpull = max (m_pull - cos_sim (WePe, W0Pg) / cos_sim (W0Pe, W0Pg) , 0)
            elif 'r' in m_incl:
                # print('using improved relative generic inclusion loss')
                m_incl_ =  float(m_incl.replace('r',''))
                # cos_incl_h = F.cosine_similarity(W_u_e_h, W_0_g_h, dim=-1)
                with torch.no_grad():
                    base_cos_incl_h = F.cosine_similarity(W_0_e_h, W_0_g_h, dim=-1)
                r = torch.clamp(m_incl_ - cos_incl_h/base_cos_incl_h, min=0.0)
                incl_terms.append(r.mean())

                all_base_generic_target_similariy += [base_cos_incl_h.detach()]
                
     
            ##
            elif 'e' in m_incl or  'ex' in m_incl:
                # the same
                with torch.no_grad():
                    m_incl_ =  F.cosine_similarity(W_0_e_h, W_0_g_h, dim=-1)
                    print('Generic-Inclusion margin based on original similarity between the target concept and the generic concept per head:\n', m_incl_)
                    
                    if 'ex' in m_incl:
                        m_incl_ *= float(m_incl.replace('ex','')) # 'e{value}' is more generalize
                        print(float(m_incl.replace('ex','')))
                    elif 'e' in m_incl:
                        m_incl_ += float(m_incl.replace('e',''))
                    
                    m_incl_ = m_incl_.clamp(min=-1.0,max=1.0) # max cosine similarity is 1.0
                    
                    print('Refined Generic-Inclusion margin based on original similarity between the target concept and the generic concept per head:\n', m_incl_)
                    
                    # print(cos_incl_h.shape,m_incl_.shape ) # [5, 77, 8]
                incl_terms.append(torch.clamp(m_incl_ - cos_incl_h, min=0.0).mean())
                
                    

            else:
                incl_terms.append(torch.clamp(float(m_incl) - cos_incl_h, min=0.0).mean())
                
            
            # # only bos
            # preserve_invalid = torch.ones((P, T), dtype=torch.bool, device=excl_term.device)
            # for i in range(P):
            #     preserve_invalid[i][0] = False

            # # preservation term
            # cos_preserve_h = F.cosine_similarity(W_u_g_h, W_0_g_h, dim=-1)
            # preserve_term =  1 - cos_preserve_h
            # preserve_terms.append(preserve_term[~preserve_invalid].mean())

            # print(f"L_preserve: {preserve_term[~preserve_invalid]}")


            # # preservation term
            cos_preserve_h = F.cosine_similarity(W_u_g_h, W_0_g_h, dim=-1)
            preserve_term =  1 - cos_preserve_h
            preserve_terms.append(preserve_term.mean())
                
                
                

            # Norm per (token, head): [B, T, H]
            norm_u = W_u_e_h.norm(dim=-1)
            norm_0 = W_0_e_h.norm(dim=-1)
            norm_terms.append(torch.abs(torch.log(norm_u) - torch.log(norm_0)).mean())
            
            
            
            
        # ---- L_w: weight-delta penalty (computed per sim_param_group) ----
        dW = (W_u - W_0)
        if sim_param_group in {"avg_token", "token"}:
            # Mean squared delta (Frobenius^2 normalized by numel)
            w_terms.append(dW.pow(2).mean())
        elif sim_param_group == "attn_head":
            num_heads = 8
            out_dim = dW.shape[0]  # rows correspond to output channels
            # if out_dim % num_heads == 0:
            head_out = out_dim // num_heads
            dW_h = dW.view(num_heads, head_out, *dW.shape[1:])  # [H, head_out, in_dim]
            # print(dW_h.shape) # 8, 160, 768
            # print(dW_h.pow(2).mean(dim=(1, 2)).shape) # 8,
            
            w_terms.append(dW_h.pow(2).mean(dim=(1, 2)).mean())  # mean over heads
            # else:
            #     # Fallback if not divisible (keeps behavior robust)
            #     w_terms.append(dW.pow(2).mean())
            
                

        matched_layers += 1

    if matched_layers == 0:
        zero = torch.tensor(0.0, device=p_e.device)
        return zero, zero, zero, zero

    L_excl = torch.stack(excl_terms).mean()
    L_incl = torch.stack(incl_terms).mean()
    L_ang = L_excl + L_incl
    L_norm = torch.stack(norm_terms).mean()
    L_w = torch.stack(w_terms).mean()
    L_preserve = torch.stack(preserve_terms).mean()


    if return_l2_baseline:
        L_push_l2 = torch.stack(l_push_l2_terms).mean()
        l_pull_l2 = torch.stack(l_pull_l2_terms).mean()
        l_preserve_l2 = torch.stack(l_preserve_l2_terms).mean()
        
        return L_excl, L_incl, L_norm, L_w, L_preserve, L_push_l2, l_pull_l2, l_preserve_l2, all_base_generic_target_similariy

    return L_excl, L_incl, L_norm, L_w, L_preserve,all_base_generic_target_similariy  #  L_ang, 

#---

    

    

    
#+++ Main_Function 
def main(args):


    

    if args.template_prompt == 'empty': 
        concept2prompt_templates = {
            'object': ["{}"],
            'style':   ["{}"]

        }
        print(f'using empty template prompts:\n {concept2prompt_templates} ')
        
    elif args.template_prompt == 'basic': 
        concept2prompt_templates = {
            'object': ["{}"],
            'style':   ["a painting in the style of {}"]
        }

    elif args.template_prompt == 'primitive': 
        concept2prompt_templates = {
            'object': ["a photo of {}"],
            'style':   ["a painting in the style of {}"]
        }

        print(f'using empty template prompts:\n {concept2prompt_templates} ')
    else:

        concept2prompt_templates = {
            'object': ["a photo of {}"],
            'style':   ["a painting in the style of {}"]
        }

        print(f'using empty template prompts:\n {concept2prompt_templates} ')
    
    if args.special_log_step is not None:
        args.special_log_step = [int(e) for e in args.special_log_step.split(',')]
        print(f'special log step: {args.special_log_step}')
    
    
        # --- W&B init (added) ---
    if args.report_to == 'wandb':
        try:
            project = 'ul_surgery'
            wandb.init(
                project=project,
                name=resolve_model_name(args),
                config=vars(args)
            )
        except Exception as _e:
            print(f"[wandb] init failed: {_e}")
    # --- end W&B init ---

    
    print(f'random seed: {args.seed}')
    rng = np.random.RandomState(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True # tested, does not increase training time
    torch.backends.cudnn.benchmark = False
    
    
    if (args.extra_forward_prob is not None and args.extra_forward_prob) > 0  or args.apply_poison:
        extra_forward_rng = np.random.RandomState(seed=args.seed + 123)
        
    
    
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    # if args.unlearn_proj_prob < 1.0:
    proj_rng = torch.Generator(device=args.device) 
    proj_rng.manual_seed(args.seed **2)

    total_grad_stats = defaultdict(list)
    if args.collect_gradient_statistics_option is not None and args.collect_gradient_statistics_option in ['dynamic','static']:
        print(f"Only collecting gradient statistics: {args.collect_gradient_statistics_option}")
        
        
        # total_grad_stats['cosine_similarities'] = []
        # total_grad_stats['norms_A'] = []    
        # total_grad_stats['norms_B'] = []    


    erase_concept = args.erase_concept

    num_inference_steps = args.num_inference_steps
    
    guidance_scale = args.guidance_scale
    negative_guidance = args.negative_guidance
    train_method=args.train_method
    max_training_step = args.max_training_step
    batch_size = args.batch_size
    # height=width=1024 # Fix to 1024 ?
    height=width=512 # I now fixed this to 512
    lr = args.lr
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    
    
    criteria = torch.nn.MSELoss()
    # torch_dtype = torch.bfloat16 : will underflow, not stable
    if args.train_precision == 'bf16':
        torch_dtype = torch.bfloat16
    elif args.train_precision == 'fp32':
        torch_dtype = torch.float32 # double training time compared to bf16


    pipe, base_unet, esd_unet = load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    
    if args.load_unet_weight_path is not None:
        print('loading UNet weight from: ', args.load_unet_weight_path)
        
        # load to esd_unet (erased_unet) only ... not for the pretrained base_unet
        esd_unet.load_state_dict(load_file(args.load_unet_weight_path), strict=False)
    
    if args.load_token_embedding_path is not None:
        
        args.load_token_embedding_path = args.load_token_embedding_path.split(';') # can be a lot
        
        for p in args.load_token_embedding_path:
            load_token_embedding(pipe.text_encoder, pipe.tokenizer,p)
    
    
    
    if args.erase_tis is not None: args.erase_tis = args.erase_tis.split(';')
    print(f'erase_ti: {args.erase_tis}')
    
    
    print(f'args.erase_ti_weights: {args.erase_ti_weights}')
    erased_unet_weights = None
    if args.erase_ti_weights is not None:
        args.erase_ti_weights = args.erase_ti_weights.split(';') # can be many paths
        assert args.erase_tis is not None and len(args.erase_tis) == len(args.erase_ti_weights), "Number of TI concepts and number of TI weight paths should be the same"
        erased_unet_weights = []
        for p in args.erase_ti_weights:
            print(f'loading erased unet weight from :{p}')
            erased_unet_weights.append(load_file(p))
    
    
    
    # pipe.scheduler.set_timesteps(100)
    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # pipe.disable_xformers_memory_efficient_attention()

    # base_unet = base_unet.eval()
    

    # esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)

    # Freeze all params in esd_unet first
    esd_unet.requires_grad_(False)

    # Now select which ones to train
    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)

    # Make sure selected ones are trainable
    for p in esd_params:
        p.requires_grad_(True)
    
    
    
    if args.weight_decay is not None:

    
        optimizer = torch.optim.Adam(esd_params, lr=lr,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(esd_params, lr=lr)

    # optimizer = torch.optim.AdamW(esd_params, lr=lr)

    # print(esd_param_names)


    # my add
    # if args.preservation_train_set == 'celeb':
    #     preservation_concepts =  torch.load('../data_root/cache/celeb/100celebrity.pt')
        
    # preservation_concepts = []
    if args.preservation_train_set == '00':
        
        preserve_cate = 'Strongly Associated'
        if args.erase_concept.lower() == 'barack obama':
            preserve_cate = 'Moderately Associated'
            # for now (Strongely Associated has too few concepts)
            
        # Hack 
        if erase_concept in  bypass_preservation_concepts:
            preservation_concepts = []
        else:
            print('still bypassing')
            #hack
            preservation_concepts = []
            # preservation_concepts =  torch.load('../data_root/data/preservation_concepts/all_pe_v3_r123.pth')[args.erase_concept.lower()][args.preservation_split][preserve_cate]
        
        
        if args.preservation_split == 'train':
            print('fixing overlap')
            
            if erase_concept in  bypass_preservation_concepts : 
                preservation_concepts = []
            else:

                test_preservation_concepts = torch.load('../data_root/data/preservation_concepts/all_pe_v3_r123.pth')[args.erase_concept.lower()]['test'][preserve_cate]
                preservation_concepts = [c for c in preservation_concepts if c not in test_preservation_concepts]
            
            
            if erase_concept in concept2SpecialEval and 'painting' in erase_concept:
                preservation_concepts = [c for c in preservation_concepts if c not in concept2SpecialEval[erase_concept]]
                
            
        
        if args.preservation_split == 'test':
            print('WARNING: optimizing on test set for preservation concepts!')
        
        # print(f"preservation_concepts: {preservation_concepts}")
    
    elif args.preservation_train_set == 'UG': 
        generic_concept = concept2generic_concept[erase_concept]
        preservation_concepts = ["",generic_concept]
        
        print(f"preservation_concepts: {preservation_concepts}")
        
    elif args.preservation_train_set == 'U':
        preservation_concepts = [""]
        print(f"preservation_concepts: {preservation_concepts}")
        
        
        

    if args.apply_gradient_projection:

        unet = esd_unet  ## or pipe.unet, depending on your context
        learnable_param_names, learnable_params = esd_param_names, esd_params


    if args.timestep_constraint is not None:
        args.lb_timestep_constraint, args.ub_timestep_constraint = map(int, args.timestep_constraint.split('-'))
        print(f'timestep constraint: {args.lb_timestep_constraint}-{args.ub_timestep_constraint}')
        constrainted_timesteps = torch.tensor([ t for t in pipe.scheduler.timesteps if t <= args.ub_timestep_constraint and t >= args.lb_timestep_constraint ]).to(args.device)
        print(f"constrainted_timesteps: {constrainted_timesteps}")
        # print(f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{erase_concept.replace(' ', '_')}-{train_method.replace('-','')}_T{args.timestep_constraint}.safetensors")
        # t_scale = args.num_inference_steps/1000
        # args.scaled_lb_timestep_constraint = int(t_scale*args.lb_timestep_constraint)
        # args.scaled_ub_timestep_constraint = int(t_scale*args.ub_timestep_constraint)
        # print(f'scaled timestep constraint: {args.scaled_lb_timestep_constraint}-{args.scaled_ub_timestep_constraint}')

    if args.decompositional_timestep_sampler is not None:
        if args.decompositional_timestep_sampler == 'avg':
            sampler_general_concept = 'a photo of person'
            sampler_stats = torch.load(f'../data_root/cache/compositional_latents/minmax_inverse_cosine_AllPerson.{sampler_general_concept}_zt_nT50.n50.bs10.seed999_{sampler_general_concept}.pt')
            print('Using avg decompositional timestep sampler')
            
        timesteps2num_inference_step = {t: i for i, t in enumerate(pipe.scheduler.timesteps.tolist())}

    with torch.no_grad():
        # get prompt embeds
        erase_embeds, null_embeds = pipe.encode_prompt(prompt=erase_concept,
                                                       device=device,
                                                       num_images_per_prompt=batch_size,
                                                       do_classifier_free_guidance=True,
                                                       negative_prompt='')
                                                 
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        
        
        
        generic_concept = concept2generic_concept[erase_concept]
        
        if args.use_refine_generic_concept:
            if generic_concept == 'person' and erase_concept!='Rihanna' and erase_concept!='Zendaya' and args.pull_specific_token!='Cs':
                generic_concept = 'the person'
            if generic_concept == 'cartoon' and erase_concept == 'Mickey Mouse':
                generic_concept = 'cartoon character'
            
        #     print('change generic concept to: ', generic_concept)
            
        
        
        
        
        ####
        erase_multiple_concepts = "CELEB" in erase_concept or "ARTIST" in erase_concept or "NAKED" in  erase_concept # 4CELEB00 ... this is a tag for using multiple concept erasing
        if erase_multiple_concepts:
            assert args.erase_tis is None # not support yet
            erase_concepts = concept2multiple_concepts[erase_concept]
            p_e_prompts = []; p_g_prompts = []; p_p_prompts = []
            p_template_prompts = []
            if 'ARTIST' in erase_concept:
                erase_concept_type = 'style'
                prompt_templates = concept2prompt_templates[erase_concept_type]
                preserve_prompt_templates = concept2preserve_prompt_templates[erase_concept_type]

                
                for ec in erase_concepts:
                    raw_erase_concept = ec.replace('a painting in the style of ','') #  "a painting in the style of Van Gogh" --> Van Gogh 
                    raw_generic_concept = generic_concept.replace('a painting in the style of ','')  # "a painting in the style of artist" --> artist
                    
                                        
                    p_e_prompts += [prompt_template.format(raw_erase_concept) for prompt_template in prompt_templates]
                    p_g_prompts += [prompt_template.format(raw_generic_concept) for prompt_template in prompt_templates]
                    p_p_prompts += [prompt_template.format(raw_generic_concept) for prompt_template in preserve_prompt_templates]
                    p_template_prompts += [prompt_template.format('') for prompt_template in prompt_templates]
                    
            elif erase_concept == '9NAKED00':
                erase_concept_type = 'object'
                prompt_templates = concept2prompt_templates[erase_concept_type]
                preserve_prompt_templates = concept2preserve_prompt_templates[erase_concept_type]


                generic_concepts = concept2multiple_generic_concepts[erase_concept]

                for ec,gc in zip(erase_concepts,generic_concepts):
                    p_e_prompts += [prompt_template.format(ec) for prompt_template in prompt_templates]
                    p_g_prompts += [prompt_template.format(gc) for prompt_template in prompt_templates]
                    p_p_prompts += [prompt_template.format(gc) for prompt_template in preserve_prompt_templates]

                    p_template_prompts += [prompt_template.format('') for prompt_template in prompt_templates]



                    
            else:
                erase_concept_type = 'object'
                prompt_templates = concept2prompt_templates[erase_concept_type]
                preserve_prompt_templates = concept2preserve_prompt_templates[erase_concept_type]
                
                for ec in erase_concepts:
                    # p_e_prompts.extend( [ec, f"a photo of {ec}", f"an image of {ec}", f"a picture of {ec}", f"a photo of a {ec}"] )
                    # p_g_prompts.extend( [generic_concept, f"a photo of {generic_concept}", f"an image of {generic_concept}", f"a picture of {generic_concept}", f"a photo of a {generic_concept}"] )
        
                    p_e_prompts += [prompt_template.format(ec) for prompt_template in prompt_templates]
                    p_g_prompts += [prompt_template.format(generic_concept) for prompt_template in prompt_templates]
                    p_p_prompts += [prompt_template.format(generic_concept) for prompt_template in preserve_prompt_templates]

                    
                    p_template_prompts += [prompt_template.format('') for prompt_template in prompt_templates]
                     
                     
                     
        else:
            if 'a painting in the style of ' in erase_concept:
                erase_concept_type = 'style'
                prompt_templates = concept2prompt_templates[erase_concept_type]
                preserve_prompt_templates = concept2preserve_prompt_templates[erase_concept_type]
                
                
                
                if args.erase_tis is None:
                    raw_erase_concept = erase_concept.replace('a painting in the style of ','') #  "a painting in the style of Van Gogh" --> Van Gogh 
                    raw_generic_concept = generic_concept.replace('a painting in the style of ','')  # "a painting in the style of artist" --> artist
                    
                    p_e_prompts = [prompt_template.format(raw_erase_concept) for prompt_template in prompt_templates]
                    p_g_prompts = [prompt_template.format(raw_generic_concept) for prompt_template in prompt_templates]
                    p_p_prompts = [prompt_template.format(raw_generic_concept) for prompt_template in preserve_prompt_templates]


                    p_template_prompts =[prompt_template.format('') for prompt_template in prompt_templates]
                    
                    
                    # p_e_prompts = [f"a painting in the style of {raw_erase_concept}", f"an artwork of {raw_erase_concept}", f"a photo in the style of {raw_erase_concept}", f"a picture in the style of {raw_erase_concept}"]
                    # p_g_prompts = [f"a painting in the style of {raw_generic_concept}", f"an artwork of {raw_generic_concept}", f"a photo in the style of {raw_generic_concept}", f"a picture in the style of {raw_generic_concept}"]
                    
                else:
                    # erase_tis = args.erase_ti.split(';') # can be more than one
                    raw_erase_concept = erase_concept.replace('a painting in the style of ','') #  "a painting in the style of Van Gogh" --> Van Gogh
                    raw_generic_concept = generic_concept.replace('a painting in the style of ','')
                    print(f'using attack token embeddings for PPP: {args.erase_tis}')
                    p_e_prompts = []; p_g_prompts = []; p_p_prompts = []
                    
                    # also adding the original
                    p_e_prompts += [prompt_template.format(raw_erase_concept) for prompt_template in prompt_templates]
                    p_g_prompts += [prompt_template.format(raw_generic_concept) for prompt_template in prompt_templates]
                    p_p_prompts += [prompt_template.format(raw_generic_concept) for prompt_template in preserve_prompt_templates]
                    
                    p_template_prompts =[prompt_template.format('') for prompt_template in prompt_templates]
                    
                    
                    for erase_ti in args.erase_tis:
                        p_e_prompts += [prompt_template.format(erase_ti) for prompt_template in prompt_templates]
                        p_g_prompts += [prompt_template.format(raw_generic_concept) for prompt_template in prompt_templates]
                        p_p_prompts += [prompt_template.format(raw_generic_concept) for prompt_template in preserve_prompt_templates]
                        # p_e_prompts += [f"a painting in the style of {erase_ti}", f"an artwork of {erase_ti}", f"a photo in the style of {erase_ti}", f"a picture in the style of {erase_ti}"]
                        # p_g_prompts += [f"a painting in the style of {raw_generic_concept}", f"an artwork of {raw_generic_concept}", f"a photo in the style of {raw_generic_concept}", f"a picture in the style of {raw_generic_concept}"]
                        p_template_prompts += [prompt_template.format('') for prompt_template in prompt_templates]
                        
                    
            else: 
                erase_concept_type = 'object'
                prompt_templates = concept2prompt_templates[erase_concept_type]
                preserve_prompt_templates = concept2preserve_prompt_templates[erase_concept_type]
                if args.erase_tis is None:
                    p_e_prompts = [prompt_template.format(erase_concept) for prompt_template in prompt_templates]
                    p_g_prompts = [prompt_template.format(generic_concept) for prompt_template in prompt_templates]
                    p_p_prompts = [prompt_template.format(generic_concept) for prompt_template in preserve_prompt_templates]

                    
                    p_template_prompts =[prompt_template.format('') for prompt_template in prompt_templates]
                     
                    # p_e_prompts = [erase_concept, f"a photo of {erase_concept}", f"an image of {erase_concept}", f"a picture of {erase_concept}", f"a photo of a {erase_concept}"]
                    # p_g_prompts = [generic_concept, f"a photo of {generic_concept}", f"an image of {generic_concept}", f"a picture of {generic_concept}", f"a photo of a {generic_concept}"]
                
                else:
                    # erase_tis = args.erase_ti.split(';') # can be more than one
                    print(f'using attack token embeddings for PPP: {args.erase_tis}')
                    p_e_prompts = []; p_g_prompts = []; p_p_prompts = []; p_template_prompts = []
                    
                    
                    p_e_prompts += [prompt_template.format(erase_concept) for prompt_template in prompt_templates]
                    p_g_prompts += [prompt_template.format(generic_concept) for prompt_template in prompt_templates]
                    p_p_prompts += [prompt_template.format(generic_concept) for prompt_template in preserve_prompt_templates]
                    
                    p_template_prompts += [prompt_template.format('') for prompt_template in prompt_templates]
                    
                    # # adding TIA tokens
                    for erase_ti in args.erase_tis:
                        p_e_prompts += [prompt_template.format(erase_ti) for prompt_template in prompt_templates]
                        p_g_prompts += [prompt_template.format(generic_concept) for prompt_template in prompt_templates]
                        p_p_prompts += [prompt_template.format(generic_concept) for prompt_template in preserve_prompt_templates]
                        p_template_prompts += [prompt_template.format('') for prompt_template in prompt_templates]
                    

                    
                    
                    # adding TIA tokens (cross pair)
                    # for erase_ti_1 in args.erase_tis:
                        
                    #     p_e_prompts += [prompt_template.format(erase_ti_1) for prompt_template in prompt_templates]
                    #     p_g_prompts += [prompt_template.format(generic_concept) for prompt_template in prompt_templates]
                    #     p_template_prompts += [prompt_template.format('') for prompt_template in prompt_templates]
                    
                        # cross pair between different TIA tokens --> better do it in the function
                        # for erase_ti_2 in args.erase_tis:
                        #     p_e_prompts += [prompt_template.format(erase_ti_1) for prompt_template in prompt_templates]

                                        
                    

                                
        print(f"p_e_prompts: {p_e_prompts}")
        print(f"p_g_prompts: {p_g_prompts}")
        print(f"p_p_prompts: {p_p_prompts}")
            
        p_e, _ = pipe.encode_prompt(prompt=p_e_prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=len(p_e_prompts)*[''])
        p_g, _ = pipe.encode_prompt(prompt=p_g_prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=len(p_g_prompts)*[''])
        p_p, _ = pipe.encode_prompt(prompt=p_p_prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=len(p_p_prompts)*[''])
        
        p_e, p_g, p_p = p_e.to(device), p_g.to(device),  p_p.to(device)
        
        
        
        # increase concentration
        # target_prompt_end_concentration_idx = None
        # if args.mask_after_eos_nth_token is not None:
        target_prompt_end_idx =  [ pipe.tokenizer(p )['input_ids'].index(pipe.tokenizer.eos_token_id) for p in p_e_prompts]
        if args.mask_after_eos_nth_token is not None:
            target_prompt_end_concentration_idx = [idx + args.mask_after_eos_nth_token for idx in target_prompt_end_idx] # offset the mask ... if 0 ... then mask right after EOS

        template_prompt_end_idx = None
        concept_token_idx = None
        if args.mask_before_concept_token or args.mask_pull_before_concept_token or args.pull_specific_token:
            # hack.. before eos token
            # print( pipe.tokenizer('Macbook')) #  [49406, 20617, 49407]
            # print( pipe.tokenizer('macbook')) # [49406, 20617, 49407]
            # print( pipe.tokenizer('laptop')) # [49406, 10464, 49407]
            # print(pipe.tokenizer(p_e_prompts))
            
            concept_token_idx = [idx -1  for idx in target_prompt_end_idx] 
            for i in range(len(p_e_prompts)):
                print(f"masking out before concept token for: {p_e_prompts[i]} | concept_token_idx: {concept_token_idx[i]}")
                

        
        if args.mask_out_template_prompt:
            template_prompt_end_idx =  [ pipe.tokenizer(p )['input_ids'].index(pipe.tokenizer.eos_token_id) for p in p_template_prompts]  # the last token before </s> is usually the concept token, we want to keep it and mask out the template part before it
            
            for i in range(len(p_e_prompts)):
                print(f"masking out template prompt for: {p_e_prompts[i]} | template_prompt_end_idx: {template_prompt_end_idx[i]}")
             
             

            
        if args.mask_pull_before_concept_token or args.pull_specific_token:
            # hack.. before eos token
            
            generic_prompt_end_idx = [ pipe.tokenizer(p )['input_ids'].index(pipe.tokenizer.eos_token_id) for p in p_g_prompts]
            
            
            generic_concept_token_idx = [idx -1  for idx in generic_prompt_end_idx] 
            for i in range(len(p_g_prompts)):
                print(f"masking out before concept token for: {p_g_prompts[i]} | concept_token_idx: {generic_concept_token_idx[i]}")  

        else:
            generic_prompt_end_idx = None 
            generic_concept_token_idx = None


        relevant_generic_concept_idx = None
        relevant_erased_concept_idx = None
        if args.pull_specific_token == 'Cs':


            erased_concepts = [erase_concept.replace('a painting in the style of','') ]
            generic_concepts = [generic_concept.replace('a painting in the style of','')]


            all_p_e_token_idx = []
            all_p_g_token_idx = []


            def find_concept_indices(tokenizer, full_prompt, concept_text):
                full_ids = tokenizer(full_prompt, add_special_tokens=True)["input_ids"]
                concept_ids = tokenizer(concept_text, add_special_tokens=False)["input_ids"]

                concept_token_idx = None
                for i in range(len(full_ids) - len(concept_ids) + 1):
                    if full_ids[i:i + len(concept_ids)] == concept_ids:
                        concept_token_idx = list(range(i, i + len(concept_ids)))
                        break

                if concept_token_idx is None:
                    raise ValueError(
                        f"Concept '{concept_text}' not found in prompt '{full_prompt}'\n"
                        f"Full tokens: {tokenizer.convert_ids_to_tokens(full_ids)}\n"
                        f"Concept tokens: {tokenizer.convert_ids_to_tokens(concept_ids)}"
                    )

                return concept_token_idx



            for p_e_prompt, p_g_prompt, erased_concept, generic_concept in zip(
                p_e_prompts, p_g_prompts, erased_concepts, generic_concepts
            ):
                p_e_token_idx = find_concept_indices(
                    pipe.tokenizer,
                    p_e_prompt,
                    erased_concept,
                )

                p_g_token_idx = find_concept_indices(
                    pipe.tokenizer,
                    p_g_prompt,
                    generic_concept,
                )

                all_p_e_token_idx.append(p_e_token_idx)
                all_p_g_token_idx.append(p_g_token_idx)

                print("\nErased prompt:", p_e_prompt)
                print("Tokens:", pipe.tokenizer.convert_ids_to_tokens(
                    pipe.tokenizer(p_e_prompt, add_special_tokens=True)["input_ids"]
                ))
                print("Erased concept:", erased_concept)
                print("Erased token idx:", p_e_token_idx)

                print("\nGeneric prompt:", p_g_prompt)
                print("Tokens:", pipe.tokenizer.convert_ids_to_tokens(
                    pipe.tokenizer(p_g_prompt, add_special_tokens=True)["input_ids"]
                ))
                print("Generic concept:", generic_concept)
                print("Generic token idx:", p_g_token_idx)


            relevant_generic_concept_idx = all_p_g_token_idx
            relevant_erased_concept_idx = all_p_e_token_idx

                
            
            
            # this will be more compliated for multiple concepts ... need to find the token idx for each concept in each prompt ... and mask out before that
            # concept_token_id = pipe.tokenizer(erase_concept)['input_ids'][-1] # using the last token of the concept, which is usually the most representative one
            # print(f"concept_token_id for {erase_concept}: {concept_token_id}")
            
            # concept_token_idx = [ pipe.tokenizer(p )['input_ids'].index(concept_token_id) for p in p_e_prompts]
            
            # for i in range(len(p_e)):
            #     print(f"masking out before concept token for: {p_e_prompts[i]} | concept_token_idx: {concept_token_idx[i]}")
                
            #     if target_prompt_end_concentration_idx is not None:
            #         print(f"also increasing concentration after target_prompt_end_concentration_idx: {target_prompt_end_concentration_idx[i]}")
               


        if args.base_concept == 'null':
            base_embeds = null_embeds
        elif args.base_concept == 'general':
            general_concept = concept2generic_concept[erase_concept]
            general_embeds, _ = pipe.encode_prompt(prompt=general_concept, device=device, num_images_per_prompt=batch_size, do_classifier_free_guidance=True, negative_prompt='')
            base_embeds = general_embeds.to(device)
        elif args.base_concept == 'neighbor':
            neighbor_concept = concept2neighbor[erase_concept]
            neighbor_embeds, _ = pipe.encode_prompt(prompt=neighbor_concept, device=device, num_images_per_prompt=batch_size, do_classifier_free_guidance=True, negative_prompt='')
            base_embeds = neighbor_embeds.to(device)
            print(f"base_concept neighbor: {neighbor_concept}")
        if args.erase_from == 'object':
            object_embeds, _ = pipe.encode_prompt(prompt="object", device=device, num_images_per_prompt=batch_size, do_classifier_free_guidance=True, negative_prompt='')
            object_embeds = object_embeds.to(device)
        elif args.erase_from == 'general':
            general_concept = concept2generic_concept[erase_concept]
            general_embeds, _ = pipe.encode_prompt(prompt=general_concept, device=device, num_images_per_prompt=batch_size, do_classifier_free_guidance=True, negative_prompt='')
            general_embeds = general_embeds.to(device)
            print(f"erase_from general concept: {general_concept}")
        elif args.erase_from == 'neighbor':
            neighbor_concept = concept2neighbor[erase_concept]
            neighbor_embeds, _ = pipe.encode_prompt(prompt=neighbor_concept, device=device, num_images_per_prompt=batch_size, do_classifier_free_guidance=True, negative_prompt='')
            neighbor_embeds = neighbor_embeds.to(device)
            print(f"erase_from neighbor concept: {neighbor_concept}")
            
        # elif args.erase_from == 'forward':
        #     if args.forward_preserve:
        #         preserve_embeds,_ = pipe.encode_prompt(prompt=preserve_concepts,
        #                                             device=device,
        #                                             num_images_per_prompt=batch_size,
        #                                             do_classifier_free_guidance=True,
        #                                             negative_prompt='')
        #         preserve2embed = { p: e for p, e in zip(preserve_concepts, preserve_embeds)}
                
        
        timestep_cond = None 
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            timestep_cond = pipe.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim).to(device=device, dtype=torch_dtype)
        

    
    
    pbar = tqdm(range(max_training_step+1), desc='Training ESD')
    losses = []
    for training_step in pbar:
        optimizer.zero_grad()
        
        if training_step % args.log_step == 0 or (args.special_log_step is not None and training_step in args.special_log_step):
            save_esd_model(esd_param_names, esd_params, args, training_step, total_grad_stats)

            # reset
            total_grad_stats = defaultdict(list)
            
        # get the noise predictions for erase concept
        pipe.unet = base_unet
        
        
        
        timesteps_list = pipe.scheduler.timesteps.tolist()
        timesteps2num_inference_step = {t: i for i, t in enumerate(pipe.scheduler.timesteps.tolist())}
        
        if getattr(args, 'timestep_sampler', None) is not None:
            
            if (args.timestep_sampler).startswith('ialpha'):
                alpha = float(args.timestep_sampler.split('ialpha')[1])
                print('applying i-alpha sampling')
                N = len(timesteps_list)
                probs = _value_based_probs_divmax(timesteps_list, alpha, is_inverse=True) # ((i+1)^alpha)/N.  ..... start with 999 -> so biased toward 0
                idx_space = np.arange(N)
                
                sampled_idx = rng.choice(idx_space, p=probs)
                timestep = timesteps_list[sampled_idx]
                num_inference_step_ = timesteps2num_inference_step[timestep]
                print(f"[timestep_sampler={args.timestep_sampler}] Ialpha={alpha:.4g} | timestep: {timestep} (idx={sampled_idx}) -> num_inference_step_: {num_inference_step_}")
                timestep = torch.tensor(timestep, device=device)

                
                
            elif (args.timestep_sampler).startswith('alpha'):
                alpha = float(args.timestep_sampler.split('alpha')[1])
                print('applying alpha sampling')

                N = len(timesteps_list)
                probs = _value_based_probs_divmax(timesteps_list, alpha) # ((i+1)^alpha)/N.  ..... start with 999 -> so biased toward 0
                idx_space = np.arange(N)
                
                sampled_idx = rng.choice(idx_space, p=probs)
                timestep = timesteps_list[sampled_idx]
                num_inference_step_ = timesteps2num_inference_step[timestep]
                print(f"[timestep_sampler={args.timestep_sampler}] alpha={alpha:.4g} | timestep: {timestep} (idx={sampled_idx}) -> num_inference_step_: {num_inference_step_}")
                timestep = torch.tensor(timestep, device=device)



        elif args.decompositional_timestep_sampler == 'avg':
            timestep =  rng.choice(sampler_stats['timesteps'], p=sampler_stats['probs'])
            num_inference_step_ = timesteps2num_inference_step[timestep]
            print(f"timestep: {timestep} - num_inference_step_: {num_inference_step_}")
            timestep = torch.tensor(timestep).to(device)
        elif args.timestep_constraint is not None:

            num_inference_step_ = rng.randint(0, len(constrainted_timesteps))
            timestep = constrainted_timesteps[num_inference_step_]
            # print(f"timestep: {timestep}")
            
            
            
            # timestep = pipe.scheduler.timesteps[run_till_timestep]
            
            # print(f"timestep: {timestep}") 
            # print(pipe.scheduler.timesteps) # reverse order : 981-1
            # print(f'effective timestep: {pipe.scheduler.timesteps[args.scaled_lb_timestep_constraint]} - {pipe.scheduler.timesteps[args.scaled_ub_timestep_constraint-1]}')

        else:
            num_inference_step_ = rng.randint(0, num_inference_steps-1)
            timestep = pipe.scheduler.timesteps[num_inference_step_] # [981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,161, 141, 121, 101,  81,  61,  41,  21,   1]
            
        
        # print(timestep)
        
        forward_seed = rng.randint(0, 2**15)
        
        # pretrained prediction
        with torch.no_grad():
            # sample xt with Pe (reverse process)
            
            
            apply_extra_forward = False
            apply_indiv_extra_forward = None
            
            if args.use_indiv_extra_forward and args.extra_forward_prob is not None and args.extra_forward_prob > 0:
                apply_indiv_extra_forward = extra_forward_rng.rand(args.batch_size) > args.extra_forward_prob

                all_forward_concepts = []
                if args.forward_general:
                    all_forward_concepts.append(concept2generic_concept[erase_concept])
                if args.forward_preserve:
                    all_forward_concepts += preservation_concepts


                forward_concepts = np.array(
                    extra_forward_rng.choice(all_forward_concepts, size=args.batch_size),
                    dtype=object
                )

                forward_concepts[~apply_indiv_extra_forward] = erase_concept
                forward_concepts = forward_concepts.tolist()
                                
                                
                print(f"forward concepts: {forward_concepts}, apply_indiv_extra_forward: {apply_indiv_extra_forward}")

                        
                # batch_size now = 1
                xt = pipe(forward_concepts , 
                    num_images_per_prompt=1, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(forward_seed), output_type='latent', height=height, width=width).images


            
            
            
            elif args.extra_forward_prob is not None and args.extra_forward_prob > 0 and extra_forward_rng.rand() < args.extra_forward_prob:
                apply_extra_forward = True
                
                all_forward_concepts = []
                if args.forward_general:
                    all_forward_concepts += [concept2generic_concept[erase_concept]]
                if args.forward_preserve:
                    all_forward_concepts += preservation_concepts
                    
                forward_concept = extra_forward_rng.choice(all_forward_concepts).item()
                    
                print(f"forward concepts: {forward_concept}, apply_extra_forward: {apply_extra_forward}")
            
                xt = pipe(forward_concept , 
                    num_images_per_prompt=batch_size, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(forward_seed), output_type='latent', height=height, width=width).images


            
            else:
                xt = pipe(erase_concept , 
                        num_images_per_prompt=batch_size, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(forward_seed), output_type='latent', height=height, width=width).images



            noise_pred_erase = pipe.unet(xt,timestep, encoder_hidden_states=erase_embeds,
                timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False,)[0]
            
            # get the noise predictions for null embeds
            noise_pred_base = pipe.unet(xt,timestep,encoder_hidden_states=base_embeds,
                timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False,)[0]

            if args.erase_from == 'uncond':
                noise_pred_erase_from = noise_pred_base
        
        
            elif args.erase_from == 'general':
                noise_pred_erase_from = pipe.unet(xt,timestep,encoder_hidden_states=general_embeds,
                    timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False)[0]
            
            elif args.erase_from == 'neighbor':
                noise_pred_erase_from = pipe.unet(xt,timestep,encoder_hidden_states=neighbor_embeds,
                    timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False)[0]

            elif args.erase_from == 'object':
                noise_pred_erase_from = pipe.unet(xt,timestep,encoder_hidden_states=object_embeds,
                    timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False)[0]

                
                
            elif args.erase_from == 'forward' and apply_extra_forward:
                print(f'erase from forward concept: {forward_concept}')
                
                forward_embeds, _ = pipe.encode_prompt(prompt=forward_concept, device=device, num_images_per_prompt=batch_size, do_classifier_free_guidance=True, negative_prompt='')



                noise_pred_erase_from = pipe.unet(xt,timestep,encoder_hidden_states=forward_embeds,
                    timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False,)[0]
                
                
            else:
                noise_pred_erase_from = noise_pred_erase
                
            if args.preservation_weight is not None and args.preservation_weight > 0:
                
                
                if batch_size == 1:
                    preservation_concept = rng.choice(preservation_concepts).item()
                    print(f"preservation_concept: {preservation_concept}")
                    preservation_embeds, _ = pipe.encode_prompt(prompt=preservation_concept, device=device, num_images_per_prompt=batch_size, do_classifier_free_guidance=True, negative_prompt='')


                                                                
                    preservation_embeds = preservation_embeds.to(device)    
                                                                
                    xt_ps = pipe(preservation_concept, 
                            num_images_per_prompt=batch_size, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(forward_seed), output_type='latent', height=height, width=width).images
                    
                    noise_pred_ps = pipe.unet(xt_ps,timestep,encoder_hidden_states=preservation_embeds,
                        timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False,)[0]
                
                    # will use this if it work
                elif batch_size > 1:
                    # print(preservation_concepts)
                    preservation_concept = rng.choice(preservation_concepts, size=(batch_size,), replace=len(preservation_concepts) < batch_size).tolist()
                    print(f"preservation_concept: {preservation_concept}")
                    preservation_embeds, _ = pipe.encode_prompt(prompt=preservation_concept, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=['']*batch_size)
                        
                        
                    xt_ps = pipe(preservation_concept, 
                            num_images_per_prompt=1, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(forward_seed), output_type='latent', height=height, width=width).images
                    noise_pred_ps = pipe.unet(xt_ps,timestep,encoder_hidden_states=preservation_embeds,
                        timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False,)[0]
                    
                ###
                

        # gradient !!!
                
        pipe.unet = esd_unet
                
        if args.preservation_weight is not None and args.preservation_weight > 0:
            # prompt=[erase_concept,preservation_concept]
            
            if batch_size == 1:
                mix_prompts = [erase_concept, preservation_concept]
            elif batch_size > 1:
                assert len(preservation_concept) == batch_size
                mix_prompts = batch_size*[erase_concept] + preservation_concept
            
            text_embeds, _ = pipe.encode_prompt(prompt=mix_prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=['']* (2*batch_size))
            total_xt = torch.cat([xt, xt_ps], dim=0)
        else: 
            text_embeds = erase_embeds
            total_xt = xt

        # have to forward once
        total_noise_pred_esd_model = pipe.unet(total_xt,timestep,encoder_hidden_states=text_embeds,
            timestep_cond=timestep_cond,cross_attention_kwargs=None,added_cond_kwargs=None,return_dict=False,)[0]
        
        
        
        # print(f"total_noise_pred_esd_model.shape: {total_noise_pred_esd_model.shape}")  # [2, 4, 64, 64]
        
        
        ### specifial negative guidance for extra forward
        if apply_indiv_extra_forward is not None:
            ng = args.batch_size*[negative_guidance]
            for i in range(args.batch_size):
                if apply_indiv_extra_forward[i]:
                    ng[i] = args.extra_forward_negative_guidance
            ng = torch.tensor(ng) 
            ng = ng.view(-1, 1, 1, 1).to(device)
            # print(f'negative guidance: {ng}')

        
        elif args.extra_forward_negative_guidance is not None and apply_extra_forward:
            ng = args.extra_forward_negative_guidance
            print(f'negative guidance: {ng}')
        else:
            ng = negative_guidance
                
        # ng = torch.tensor(args.batch_size*[ng]) 
        # ng = ng.view(-1, 1, 1, 1).to(device)
        # print(f'ng.shape: {ng.shape}')
        
        # print(total_noise_pred_esd_model.shape)  # [2, 4, 64, 64]
        
        #+++ Loss_Computation
        if args.preservation_weight is not None and args.preservation_weight > 0: 
            noise_pred_esd_model, noise_pred_ps_esd_model = total_noise_pred_esd_model.chunk(2, dim=0)
            # print(noise_pred_ps_esd_model)
            # print(noise_pred_esd_model.shape, noise_pred_ps_esd_model.shape) [1, 4, 64, 64]
            
            # change to float
            noise_pred_esd_model = noise_pred_esd_model.float()
            noise_pred_ps_esd_model = noise_pred_ps_esd_model.float()
            noise_pred_ps = noise_pred_ps.float()
            noise_pred_erase = noise_pred_erase.float()
            noise_pred_erase_from = noise_pred_erase_from.float()

            
            
            unlearn_loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (ng*(noise_pred_erase - noise_pred_base))) 
            preservation_loss = criteria(noise_pred_ps_esd_model, noise_pred_ps)
            
            
        else:
            
            
            
            
            noise_pred_esd_model = total_noise_pred_esd_model

            # change to float 
            noise_pred_erase = noise_pred_erase.float()
            noise_pred_esd_model = noise_pred_esd_model.float()
            noise_pred_erase_from = noise_pred_erase_from.float()
            noise_pred_base = noise_pred_base.float()
            
            
            
            if args.neg_guidance_method == 'ccfg':
                
                assert args.base_concept == 'null'
                noise_pred_null = noise_pred_base
                
                
                noise_contrast = noise_pred_null
                noise_contrast += guidance_scale*(noise_pred_erase_from-noise_pred_null)
                tau = 0.2
                l2norm = tau * ((noise_pred_erase - noise_pred_null) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                # print(l2norm.shape)
                noise_contrast -= negative_guidance * (noise_pred_erase - noise_pred_null) * 2 * (torch.exp(-l2norm) / (1 + torch.exp(-l2norm)))

                unlearn_loss = criteria(noise_pred_esd_model, noise_contrast) 

        
            else:
            
                unlearn_loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (ng*(noise_pred_erase - noise_pred_base))) 
            preservation_loss = torch.tensor(0.0).to(device)
            
            
            # print(f"noise_pred_esd_model.shape: {noise_pred_esd_model.shape}") # [bs, 4, 64, 64]
            
        optimizer.zero_grad(set_to_none=True)

        
        loss_print_logs = {}
        
        preservation_weight = args.preservation_weight if args.preservation_weight is not None else 0.0

        
        loss_print_logs['unlearn_loss'] = unlearn_loss.item()
        loss_print_logs['preservation_loss'] = preservation_loss.item()
        
        
        if args.aei_loss_weight > 0.0:
            
            # print('loss computation: angular exclusion-inclusion loss enabled')
            
            # print(p_e.shape, p_g.shape)
            
            if args.use_l2_loss:
                print('test')
                # L_excl, L_incl, L_ang, L_norm, L_w, L_preserve, L_push_l2, l_pull_l2, l_preserve_l2
                # replace Lang (cosine) with the L2
                _,_,a_norm_loss,weight_modification_loss, _, a_excl_loss,a_incl_loss,a_preserve_loss,all_base_generic_target_similariy = compute_angular_exclusion_inclusion_loss(
                    unet_u=esd_unet, 
                    unet_0=base_unet,
                    p_e=p_e,
                    p_g=p_g,
                    p_p=p_p,

                    m_excl=args.ang_excl_margin if args.final_ang_excl_margin is None else args.final_ang_excl_margin,
                    m_incl=args.ang_incl_margin,
                    sim_param_group=args.sim_param_group,
                    max_token_seq_len=args.ang_loss_max_token_seq_len,
                    
                    # for advance experiment
                    unet_u_tis=erased_unet_weights,
                    push_pair_option=args.ang_push_pair_option,
                    n_prompt_template=len(concept2prompt_templates[erase_concept_type]),
                    all_pair_aggr_concept_option=args.ang_all_pair_aggr_concept_option,
                    
                    aggr_feature_option = args.ang_aggr_feature_option,
                    
                    compute_token_option = args.ang_compute_token_option,
                    aggr_token_option = args.ang_aggr_token_option,
                    
                    template_prompt_end_idx = template_prompt_end_idx if args.mask_out_template_prompt else None,
                    
                    apply_mask_before_concept_token = args.mask_before_concept_token, # have to specified because concept_token_idx used in Mask Generic Pull too
                    concept_token_idx = concept_token_idx,
                    # concept_token_idx = concept_token_idx if args.mask_before_concept_token else None,
                    
                    
                    target_prompt_end_concentration_idx=target_prompt_end_concentration_idx if args.mask_after_eos_nth_token is not None else None,
                    rescale_from_masking = args.rescale_from_masking,   
                    
                    
                    generic_concept_token_idx=generic_concept_token_idx if (args.mask_pull_before_concept_token or args.pull_specific_token) else None,
                    generic_prompt_end_idx = generic_prompt_end_idx,

                    pull_specific_token = args.pull_specific_token,
                    preserve_specific_token=args.preserve_specific_token,
                    return_l2_baseline = True
                    # mask_after_eos_nth_token = target_prompt_end_idx if args.mask_after_eos_nth_token is not None else None,
                    
        
                )
                print(a_excl_loss)
            
            
            
            else:
                a_excl_loss,a_incl_loss,a_norm_loss,weight_modification_loss, a_preserve_loss,all_base_generic_target_similariy = compute_angular_exclusion_inclusion_loss(
                    unet_u=esd_unet, 
                    unet_0=base_unet,
                    p_e=p_e,
                    p_g=p_g,
                    p_p=p_p,

                    m_excl=args.ang_excl_margin if args.final_ang_excl_margin is None else args.final_ang_excl_margin,
                    m_incl=args.ang_incl_margin,
                    sim_param_group=args.sim_param_group,
                    max_token_seq_len=args.ang_loss_max_token_seq_len,
                    
                    # for advance experiment
                    unet_u_tis=erased_unet_weights,
                    push_pair_option=args.ang_push_pair_option,
                    n_prompt_template=len(concept2prompt_templates[erase_concept_type]),
                    all_pair_aggr_concept_option=args.ang_all_pair_aggr_concept_option,
                    
                    aggr_feature_option = args.ang_aggr_feature_option,
                    
                    compute_token_option = args.ang_compute_token_option,
                    aggr_token_option = args.ang_aggr_token_option,
                    
                    apply_mask_before_concept_token = args.mask_before_concept_token, # have to specified because concept_token_idx used in Mask Generic Pull too
                    concept_token_idx = concept_token_idx,
                    # concept_token_idx = concept_token_idx if args.mask_before_concept_token else None,
                    
                    template_prompt_end_idx = template_prompt_end_idx if args.mask_out_template_prompt else None,
                    
                    target_prompt_end_concentration_idx=target_prompt_end_concentration_idx if args.mask_after_eos_nth_token is not None else None,
                    rescale_from_masking = args.rescale_from_masking,
                    # mask_after_eos_nth_token = target_prompt_end_idx if args.mask_after_eos_nth_token is not None else None,
                    generic_concept_token_idx=generic_concept_token_idx if (args.mask_pull_before_concept_token or args.pull_specific_token) else None,

                    generic_prompt_end_idx=generic_prompt_end_idx,
                    pull_specific_token=args.pull_specific_token,
                    preserve_specific_token=args.preserve_specific_token,

                    relevant_erased_concept_idx=relevant_erased_concept_idx,
                    relevant_generic_concept_idx=relevant_generic_concept_idx
        
                )
            if args.only_save_base_sim:
                base_dir = 'similariy_analysis/prelim_sim'



                if args.template_prompt == 'empty':
                    base_file_name += '-TEe'
                elif args.template_prompt == 'basic':
                    base_dir += '-TEb'
                elif args.template_prompt == 'primitive':
                    base_dir += '-TEp'



                if args.pull_specific_token is not None:
                    base_dir += f'-{args.pull_specific_token}'  

                elif args.pull_specific_token is None and args.mask_pull_before_concept_token:
                    base_dir += f'-pc'  

                os.makedirs(base_dir,exist_ok=True)


                save_path__ = osp.join(base_dir, f'{args.erase_concept}.safetensors')
                torch.save(all_base_generic_target_similariy,save_path__)
                print(f'save base similarity to {save_path__}')
                print(all_base_generic_target_similariy)

                exit()

            

            # print(erase_embeds[0:1,:,:].shape, general_embeds[0:1,:,:].shape)
            
            
            aei_loss = (args.ang_excl_loss_weight*a_excl_loss) + (args.ang_incl_loss_weight* a_incl_loss) + args.ang_norm_loss_weight*a_norm_loss
            
            if args.final_ang_preserve_loss_weight is not None:
                aei_loss += args.final_ang_preserve_loss_weight * a_preserve_loss
            else:
                aei_loss += args.ang_preserve_loss_weight* a_preserve_loss
            # aei_loss += a_mid_loss
            
            
            loss_print_logs['ang_excl_loss'] = a_excl_loss.item() * args.ang_excl_loss_weight
            loss_print_logs['ang_incl_loss'] = a_incl_loss.item() * args.ang_incl_loss_weight
            loss_print_logs['ang_norm_loss'] = a_norm_loss.item()
            # loss_print_logs['ang_mid_loss'] = a_mid_loss.item()
            loss_print_logs['aei_loss'] = aei_loss.item()
            loss_print_logs['ang_preserve_loss'] = a_preserve_loss.item()  # * args.ang_preserve_loss_weight
            
            # initialize with aei loss
            total_loss = args.aei_loss_weight*aei_loss 
                
            # generic mapping/ neg guidance/unlearn loss
            # loss_print_logs['unlearn_loss'] = unlearn_loss.item() ... already logged above
            total_loss += args.generic_loss_weight * unlearn_loss
            
            # preservation loss
            # loss_print_logs['preservation_loss'] = preservation_loss.item() .... already logged above
            
            total_loss += preservation_weight * preservation_loss
            # total_loss += preservation_weight * preservation_loss
            
            
            
            
            # l2 weight modification loss (inconsistent for now)
            if args.weight_modification_weight is not None and args.weight_modification_weight > 0.0:
                loss_print_logs['weight_modification_loss'] = weight_modification_loss.item()
                total_loss += args.weight_modification_weight*weight_modification_loss
                

            
        else:
            if args.preservation_weight_option == 'convex':
                total_loss = (1.0 - preservation_weight) * unlearn_loss + preservation_weight * preservation_loss
            else:
                total_loss = unlearn_loss + preservation_weight * preservation_loss
                
                
                

                
        loss_print_logs['total_loss'] = total_loss.item()
        
        
        print(" | ".join([f"{k}: {v:.6f}" for k,v in loss_print_logs.items()]))
        

        total_loss.backward()
        
        
        print(f'total loss:',total_loss)
        


        optimizer.step()
        
        
        #---

#+++ WandB logging
        # --- W&B log (after step) ---
        if args.report_to == 'wandb':
            try:
                wandb.log({
                    "unlearning_loss": float(unlearn_loss.detach().item()),
                    "preservation_loss": float(preservation_loss.detach().item()),
                    "aei_loss": float(aei_loss.detach().item()) if args.aei_loss_weight > 0.0 else None,
                    "ang_excl_loss": float(a_excl_loss.detach().item()) * args.ang_excl_loss_weight if args.aei_loss_weight > 0.0 else None,
                    "ang_incl_loss": float(a_incl_loss.detach().item()) * args.ang_incl_loss_weight if args.aei_loss_weight > 0.0 else None,
                    "ang_preserve_loss": float(a_preserve_loss.detach().item())  if args.aei_loss_weight > 0.0 else None,
                    "ang_norm_loss": float(a_norm_loss.detach().item()) if args.aei_loss_weight > 0.0 else None,
                    "weight_modification_loss": float(weight_modification_loss.detach().item())  if args.aei_loss_weight > 0.0 else None,
                    "total_weighted_loss": float((total_loss).detach().item()) if args.preservation_weight is not None else float(unlearn_loss.detach().item()),
                    "timestep": int(timestep.detach().cpu().item()),
                    "training_step": int(training_step),
                    
                    # "ang_mid_loss": float(a_mid_loss.detach().item()) if args.aei_loss_weight > 0.0 else None,
                    
                }, step=int(training_step))
            except Exception as _e:
                print(f"[wandb] log failed: {_e}")
                # --- end W&B log ---
#---
#---


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    
    
    main(args)
    
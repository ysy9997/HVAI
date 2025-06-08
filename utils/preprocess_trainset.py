"""
https://dacon.io/competitions/official/236493/talkboard/413868
1) 'K5_3세대_하이브리드_2020_2022'와 'K5_하이브리드_3세대_2020_2023'는 동일 클래스로 간주하여 평가됩니다.
2) '디_올뉴니로_2022_2025'와 '디_올_뉴_니로_2022_2025'는 동일 클래스로 간주하여 평가됩니다.
3) '718_박스터_2017_2024'와 '박스터_718_2017_2024'는 동일 클래스로 간주하여 평가됩니다.
4) 'RAV4_2016_2018'와 '라브4_4세대_2013_2018'는 동일 클래스로 간주하여 평가됩니다.
5) 'RAV4_5세대_2019_2024'와 '라브4_5세대_2019_2024'는 동일 클래스로 간주하여 평가됩니다.
"""
import os
import shutil
import argparse
from tqdm import tqdm


MERGE_DICT = {
    'K5_3세대_하이브리드_2020_2022':'K5_하이브리드_3세대_2020_2023',
    '디_올뉴니로_2022_2025':'디_올_뉴_니로_2022_2025',
    '718_박스터_2017_2024':'박스터_718_2017_2024',
    'RAV4_2016_2018':'라브4_4세대_2013_2018',
    'RAV4_5세대_2019_2024':'라브4_5세대_2019_2024'
}
NOISY_IMAGES = {
    '더_뉴_QM6_2024_2025_0040.jpg', 'A8_D5_2018_2023_0084.jpg', '파나메라_2010_2016_0000.jpg', '타이칸_2021_2025_0065.jpg', 
    '머스탱_2015_2023_0086.jpg', 'RAV4_5세대_2019_2024_0020.jpg', '콰트로포르테_2017_2022_0074.jpg', '더_뉴_파사트_2012_2019_0067.jpg', 
    'F150_2004_2021_0018.jpg', '더_기아_레이_EV_2024_2025_0078.jpg', '4시리즈_F32_2014_2020_0027.jpg', 'Q50_2014_2017_0031.jpg', 
    '컨티넨탈_GT_3세대_2018_2023_0007.jpg', '911_992_2020_2024_0030.jpg', '뉴_CC_2012_2016_0001.jpg', 'S_클래스_W223_2021_2025_0071.jpg', 
    '5시리즈_G60_2024_2025_0056.jpg', 'Q30_2017_2019_0075.jpg', '베뉴_2020_2024_0005.jpg', 'GLE_클래스_W167_2019_2024_0068.jpg', 
    '뉴_G80_2025_2026_0043.jpg', '그랜드_체로키_WL_2021_2023_0018.jpg', '더_뉴_그랜드_스타렉스_2018_2021_0079.jpg', '프리우스_4세대_2019_2022_0052.jpg', 
    'E_클래스_W212_2010_2016_0069.jpg', '6시리즈_GT_G32_2018_2020_0018.jpg', '뉴_ES300h_2013_2015_0000.jpg', 'S_클래스_W223_2021_2025_0008.jpg', 
    'GLS_클래스_X167_2020_2024_0013.jpg', 'GLB_클래스_X247_2020_2023_0008.jpg', '뉴_G80_2025_2026_0042.jpg', '레니게이드_2019_2023_0041.jpg', 
    'X3_G01_2022_2024_0029.jpg', '레인지로버_4세대_2018_2022_0048.jpg', 'K5_2세대_2016_2018_0007.jpg', '싼타페_TM_2019_2020_0009.jpg', 
    'E_클래스_W212_2010_2016_0022.jpg', '카이엔_PO536_2019_2023_0054.jpg', '더_뉴_그랜드_스타렉스_2018_2021_0078.jpg', 'X4_F26_2015_2018_0068.jpg', 
    '2시리즈_액티브_투어러_U06_2022_2024_0004.jpg', '아반떼_MD_2011_2014_0082.jpg', 'EQA_H243_2021_2024_0063.jpg', '아베오_2012_2016_0052.jpg', 
    '티볼리_에어_2016_2019_0047.jpg', '4시리즈_G22_2024_2025_0031.jpg', '레인지로버_스포츠_2세대_2018_2022_0014.jpg', 'ES300h_7세대_2019_2026_0028.jpg', 
    '7시리즈_F01_2009_2015_0044.jpg', '박스터_718_2017_2024_0011.jpg', 'X7_G07_2019_2022_0052.jpg', '아반떼_MD_2011_2014_0009.jpg', 
    '뉴_SM5_임프레션_2008_2010_0033.jpg', '7시리즈_F01_2009_2015_0029.jpg', '익스플로러_2016_2017_0072.jpg', 'Q5_FY_2021_2024_0032.jpg', 
    '마칸_2019_2021_0035.jpg', 'G_클래스_W463b_2019_2025_0030.jpg', '아반떼_N_2022_2023_0064.jpg', '파나메라_2010_2016_0036.jpg', 
    'SM7_뉴아트_2008_2011_0053.jpg', 'CLS_클래스_C257_2019_2023_0021.jpg', 'G_클래스_W463_2009_2017_0011.jpg', 'XF_X260_2016_2020_0023.jpg', 
    '더_뉴_아반떼_2014_2016_0031.jpg', '글래디에이터_JT_2020_2023_0075.jpg', '더_뉴_그랜드_스타렉스_2018_2021_0080.jpg', '3시리즈_F30_2013_2018_0036.jpg', 
    'K3_2013_2015_0045.jpg', 'Q30_2017_2019_0074.jpg', 'G_클래스_W463b_2019_2025_0049.jpg', '레인지로버_스포츠_2세대_2018_2022_0017.jpg', 
    '더_뉴_K3_2세대_2022_2024_0001.jpg', '레이_2012_2017_0063.jpg', 'C_클래스_W204_2008_2015_0068.jpg', '레인지로버_5세대_2023_2024_0030.jpg', 
    '7시리즈_G11_2016_2018_0040.jpg', '카이엔_PO536_2019_2023_0035.jpg', '아반떼_MD_2011_2014_0081.jpg', '뉴_A6_2012_2014_0046.jpg', 
    'Q7_4M_2020_2023_0011.jpg', '뉴_CC_2012_2016_0002.jpg', '911_992_2020_2024_0006.jpg', 'K5_3세대_2020_2023_0081.jpg', 
    '5시리즈_G60_2024_2025_0010.jpg', '아반떼_N_2022_2023_0035.jpg', '더_뉴_코나_2021_2023_0081.jpg'
}

def merge_redundant_classes(args):
    for dir1, dir2 in MERGE_DICT.items():
        # check validity of all directories
        assert os.path.exists(f"{args.imdir}/{dir1}") and os.path.isdir(f"{args.imdir}/{dir1}"), f"Invalid directory: {args.imdir}/{dir1}"
        assert os.path.exists(f"{args.imdir}/{dir2}") and os.path.isdir(f"{args.imdir}/{dir2}"), f"Invalid directory: {args.imdir}/{dir2}"

        # check number of files in each directories
        print("# files in {}: {}".format(dir1, len(os.listdir(f"{args.imdir}/{dir1}"))))
        print("# files in {}: {}".format(dir2, len(os.listdir(f"{args.imdir}/{dir2}"))))

        # move all files from dir1 to dir2
        print(f"Moving all files from {dir1} to {dir2}")
        for fn in os.listdir(f"{args.imdir}/{dir1}"):
            shutil.move(f"{args.imdir}/{dir1}/{fn}", f"{args.imdir}/{dir2}/{fn}")
        
        # remove dir1
        print(f"Deleting folder {dir1}")
        shutil.rmtree(f"{args.imdir}/{dir1}")

        # check number of files in each directories
        print("# files in {}: {}\n".format(dir2, len(os.listdir(f"{args.imdir}/{dir2}"))))

def move_noisy_images(args):
    make_list = os.listdir(args.imdir)
    if "_cat_or_dog" in make_list:
        assert len(make_list) == 392, f"Wrong number of classes:{len(make_list)} (must be 391 + _cat_or_dog). Make sure to run merge_redundant_classes() first."
    else:
        assert len(make_list) == 391, f"Wrong number of classes:{len(make_list)} (must be 391). Make sure to run merge_redundant_classes() first."
    os.makedirs(args.imdir + '/_cat_or_dog', exist_ok=True)
    
    move_counter = 0
    for make in tqdm(make_list):
        if make == "_cat_or_dog": continue
        for fn in os.listdir(f"{args.imdir}/{make}"):
            if fn in NOISY_IMAGES:
                move_counter += 1
                shutil.move(f"{args.imdir}/{make}/{fn}", f"{args.imdir}/_cat_or_dog/{fn}")
                print(f"{move_counter}: {args.imdir}/{make}/{fn} -> {args.imdir}/_cat_or_dog/{fn}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('imdir', type=str)
    parser.add_argument('--merge_redundant_classes', action='store_true')
    parser.add_argument('--move_noisy_images', action='store_true')
    args = parser.parse_args()
    mergedFlag, movedFlag = False, False
    if args.merge_redundant_classes:
        merge_redundant_classes(args)
        mergedFlag = True
    if args.move_noisy_images:
        move_noisy_images(args)
        movedFlag = True
    if not (mergedFlag or movedFlag):
        print("No change made! Set at least one of '--merge_redundant_classes' or '--move_noisy_images' flags")
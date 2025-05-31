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


MERGE_DICT = {
    'K5_3세대_하이브리드_2020_2022':'K5_하이브리드_3세대_2020_2023',
    '디_올뉴니로_2022_2025':'디_올_뉴_니로_2022_2025',
    '718_박스터_2017_2024':'박스터_718_2017_2024',
    'RAV4_2016_2018':'라브4_4세대_2013_2018',
    'RAV4_5세대_2019_2024':'라브4_5세대_2019_2024'
}

def main(args):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('imdir', type=str)
    args = parser.parse_args()
    main(args)
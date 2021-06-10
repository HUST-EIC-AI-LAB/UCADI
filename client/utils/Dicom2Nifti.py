#  Copyright (c) 2020. Hanchen Wang, hc.wang96@gmail.com

import os, argparse, dicom2nifti, dicom2nifti.settings as settings
from tqdm import tqdm


if __name__ == '__main__':

	parser = argparse.ArgumentParser('Dicom2Nifti')
	parser.add_argument('--dicom', type=str, default=None, help='Directory for DICOM Image Series/Directories')
	parser.add_argument('--output', type=str, default=None, help='Output NIFTI Filename/Directory')
	parser.add_argument('--batch', action='store_true', default=False, help='Batch Generation')

	settings.disable_validate_slice_increment()
	settings.enable_resampling()
	settings.set_resample_spline_interpolation_order(1)
	settings.set_resample_padding(-1000)

	'''Ref: https://dicom2nifti.readthedocs.io/en/latest/_modules/dicom2nifti/convert_dicom.html#dicom_series_to_nifti'''
	'''Ref: https://dicom2nifti.readthedocs.io/en/latest/readme.html#from-python'''
	args = parser.parse_args()
	if args.batch:
		os.mkdir(args.output) if not os.path.exists(args.output) else None
		for dir_ in tqdm(os.listdir(args.dicom)):
			# dicom2nifti.convert_directory(args.dicom, args.output)
			intputd = os.path.join(args.dicom, dir_)
			outputf = os.path.join(args.output, dir_)
			dicom2nifti.convert_dicom.dicom_series_to_nifti(intputd, outputf)
	else:
		dicom2nifti.convert_dicom.dicom_series_to_nifti(args.dicom, args.output)

Place the private 'citation-network.csv' into the folder

'CourtCases/default/raw'

Then run the preprocessing files in their numbered order.

You also have to put in the database credentials in the files

'./preprocessing/_1_generate_subsampled_citations_pkl.py'
'./preprocessing/-_2_build_quintuple_dict.py'

and set up an SSH connection to the ETH network as described
in the comments of the code.
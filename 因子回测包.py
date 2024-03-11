import pandas as pd
import numpy as np


def align_factor(factors):
	index = factors[0].index
	columns = factors[0].columns
	result = [factors[0]]
	for i in factors[1:]:
		to_append = i.reindex(index=index,method='pad')
		to_append = to_append.reindex(columns=columns)
		result.append(to_append)
	return result


def factor_correlation(factor1,factor2,min_valid_num=0):

	factor1.replace({None:np.nan},inplace=True)
	factor1=factor1.astype(float)
	factor2.replace({None:np.nan},inplace=True)
	factor2=factor2.astype(float)	
	
	factor1_sum = factor1.notnull().sum(axis=1)
	factor1.loc[factor1_sum<min_valid_num,:]=np.nan
	factor2_sum = factor2.notnull().sum(axis=1)
	factor2.loc[factor2_sum<min_valid_num,:]=np.nan
	
	pearson_corr=factor1.corrwith(factor2,axis=1)	
	spearman_corr=factor1.rank(axis=1).corrwith(factor2.rank(axis=1),axis=1)
	
	return pearson_corr,spearman_corr

def factor_group(factor,split_method='average',split_num=5,industry_factor=None,limit_df=None):
	if limit_df is not None:
		[factor,limit_df] = align_factor([factor,limit_df])
		limit_df = limit_df.fillna(value=True).astype('bool')
		factor = factor[limit_df]
	if industry_factor is None:
		industry_factor = pd.DataFrame(index=factor.index,columns=factor.columns,data='Market')
		industry_factor = industry_factor[factor.notnull()].astype('object')
	else:
		[factor,industry_factor] = align_factor([factor,industry_factor])
		industry_factor = industry_factor.astype('object')
		industry_factor = industry_factor.fillna(value='others')
		industry_factor = industry_factor[factor.notnull()]
	data = pd.DataFrame(index=pd.MultiIndex.from_product([factor.index,factor.columns],names=['date','asset']))
	data['group'] = industry_factor.stack()
	data['factor'] = factor.stack()
	data = data.dropna(subset=['group'])
	data_factor_array = data['factor'].values
	data_final_split = np.full((len(data_factor_array),),np.nan)
	grouper = [data.index.get_level_values('date'),'group']
	data_groupby = data.groupby(grouper)
	data_groupby_indices = data_groupby.indices
	data_groupby_indices = list(data_groupby_indices.values())
	
	def auxilary_get_split_array(data_factor_array,data_final_split,data_groupby_indices,split_method,split_num):
		def quantile_split(_this_split_result,_this_array,_split_percentile):
			split_value = np.nanpercentile(_this_array,_split_percentile)
			split_value[0] -= 1
			split_value[-1] += 1
			for i in range(len(split_value)-1):
				_this_split_result[(_this_array<=split_value[i+1])&(_this_array>split_value[i])] = i
			return _this_split_result
			
		if split_method=='average':
			split_percentile = np.linspace(0,100,split_num+1)
		elif split_method=='largest_ratio':
			split_percentile = np.array([0,100-split_num*100,100])
		elif split_method=='smallest_ratio':
			split_percentile = np.array([0,split_num*100,100])
		
		for this_group_place in range(len(data_groupby_indices)):
			this_indice_place = data_groupby_indices[this_group_place]
			this_factor_array = data_factor_array[this_indice_place]
			this_split_result = data_final_split[this_indice_place]
			# if split_method in ['average','largest','smallest']:
			if split_method =='average':
				this_data_final_split = quantile_split(this_split_result,this_factor_array,split_percentile)
				data_final_split[this_indice_place] = this_data_final_split
			elif split_method=='smallest':
				this_factor_array_sort = np.sort(this_factor_array[~np.isnan(this_factor_array)])
				split_value = this_factor_array_sort[min(len(this_factor_array_sort)-1,split_num-1)]			
				if len(split_value)>0:
					this_split_result[this_factor_array<=split_value]=0
					this_split_result[this_factor_array>split_value]=1
					data_final_split[this_indice_place] = this_split_result
			elif split_method=='largest':
				this_factor_array_sort = np.sort(this_factor_array[~np.isnan(this_factor_array)])[::-1]
				split_value = this_factor_array_sort[min(len(this_factor_array_sort)-1,split_num-1)]
				if len(split_value)>0:
					this_split_result[this_factor_array<split_value] = 0
					this_split_result[this_factor_array>=split_value] = 1
					data_final_split[this_indice_place] = this_split_result
		return data_final_split
	
	data_final_split = auxilary_get_split_array(data_factor_array,data_final_split,data_groupby_indices,split_method,split_num)	
	data.loc[:,'factor'] = data_final_split	
	final_data = data['factor'].unstack().reindex(index=factor.index,columns=factor.columns)
	
	return final_data
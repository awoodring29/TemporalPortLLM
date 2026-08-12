import json
import warnings
import os
import matplotlib.pyplot as plt

def read_file(dir,t_metric):
    with open(dir,'r') as file:
        try:
            data=json.load(file)
        except Exception as e:
            warnings.warn(f"Skipping file {dir}. Trying to load produced error: {e}.")
            return False
        try:
            metric='cost' if not data['compare_zero'] else 'benefit'
            if metric != t_metric:
                return False
            figname=f"{data['dataset']}_{data['t']}_{data['run']}_{metric}"
        except KeyError as e:
            warnings.warn(f"Skipping {dir} because of the following key error: {e}.")
        return data,figname

# def get_cos_sim_reciprocal(data):
#     vals=[]
#     ci95s=[]
#     delta_ft_norm=data['delta_ft_norm']
#     for alpha in data['data'].keys():
#         inner_prods=data['data'][alpha]['vg_samples']
#         grad_norms=data['data'][alpha]['grad_norm_samples']
#         samples=[(gn*delta_ft_norm)/abs(inner) for inner,gn in zip(inner_prods,grad_norms)]
#         mean=sum(samples)/len(samples)
#         std=(sum([(val-mean)**2 for val in samples])/(len(samples)-1))**0.5
#         vals.append(mean)
#         ci95s.append(1.96*std/(len(samples)**0.5))
#     return vals,ci95s

def format_num(num,precision=2):
    # format numbers for annotations
    s=format(num,f".{precision}e")
    if "-" in s:
        s=s.split("e")[0]+"e-"+s.split("e")[1].removeprefix("-").removeprefix("0")
    else:
        s=s.split("e")[0]+"e"+s.split("e")[1].removeprefix("+").removeprefix("0")
    s=s.replace("e",r"\times 10^{")+"}"
    return s

def weighted_stats(samples,weights):
    # weighted mean and std across batches based on number of tokens per batch
    mean=sum([w*s for w,s in zip(weights,samples)])/sum(weights)
    std=((sum([w*(s-mean)**2 for w,s in zip(weights,samples)]))/(sum(weights)*(len(weights)-1)/(len(weights))))**0.5
    return mean,std

def plot_vg(start_dir,dataset,metric):
    # plot 1-D slice for single repetition
    dset_names={'boolq':'BoolQ',
                'winogrande':'WinoGrande',
                'arc_easy':'ARC Easy',
                'arc_challenge':'ARC Challenge',
                'humaneval':'HumanEval',
                'gsm8k':'GSM8k'}
    for dir in os.listdir(start_dir):
        if not read_file(os.path.join(start_dir,dir),metric):
            continue
        data,name=read_file(os.path.join(start_dir,dir),metric)   
        if( not data['dataset']==dataset):
            continue
        else:
            fig,ax=plt.subplots(layout='constrained',figsize=(6.4,3.5))
            ax2=ax.twinx()
            res=data['data']
            alphas=[]
            vg_means=[]
            vg_95ci=[]
            loss_means=[]
            loss_95ci=[]
            gn_means=[] # gn=grad norm
            gn_95ci=[]
            weights=data['token_lengths']
            for alpha,val in res.items():
                alphas.append(float(alpha))
                vg_mean,vg_std=weighted_stats(val['vg_samples'],weights)
                vg_means.append(float(vg_mean))
                vg_95ci.append(1.96*vg_std/(len(val['vg_samples'])**0.5))
                loss_mean,loss_std=weighted_stats(val['loss_samples'],weights)
                loss_means.append(float(loss_mean))
                loss_95ci.append(1.96*loss_std/(len(val['loss_samples'])**0.5))
                gn_mean,gn_std=weighted_stats(val['grad_norm_samples'],weights)
                gn_means.append(gn_mean)
                gn_95ci.append(1.96*gn_std/(len(val['grad_norm_samples'])**0.5))
            subscript="c" if metric=='cost' else 'b'
            norm_prod_label=r"$\|\nabla \ell_{\theta_t}(\phi^{(\alpha)})\|\|\Delta_t^\text{(ft)}\|$" if metric=='cost' else r"$\|\nabla \ell_{\theta_t}((1-\alpha)\phi_0)\||\phi_0\|$"
            if data.get('rand_dir',False):
                subscript="n"
                norm_prod_label=r"$\|\nabla \ell_{\theta_t}(\phi_t+\alpha\xi)\|\|\Delta_t^\text{(ft)}\|$"
            delta_ft_norm=data['delta_ft_norm']
            norms=[delta_ft_norm*gn for gn in gn_means]
            norm_95ci=[delta_ft_norm*ci for ci in gn_95ci]
            ax.errorbar(alphas,loss_means,loss_95ci,color='g',linestyle='dashed',label=r"$g_"+subscript+r"(\alpha)$",linewidth=3)
            ax2.errorbar(alphas,norms,norm_95ci,color='b',linestyle='dotted',label=norm_prod_label,linewidth=3)
            ax.errorbar(alphas,vg_means,vg_95ci,color='r',linestyle='solid',label=r"$\frac{d}{d\alpha} g_"+subscript+r"(\alpha)$",linewidth=3)
            ax.set_xlabel(r"$\alpha$",fontsize=26)
            ax2.set_ylabel(r"Norm Product",fontsize=26,color='blue')
            ax.tick_params(axis='both',labelsize=20)
            ax2.tick_params(axis='both',labelsize=20)
            m_name=metric[0].upper()+ metric[1:]
            m_name=f"{dset_names[dataset]} t={data['t']} {m_name}"
            # plot cost or benefit
            diff=loss_means[-1]-loss_means[0]
            if metric=='cost':
                level=min([l-ci for l,ci in zip(loss_means,loss_95ci)])*0.95
                ax.text(0.1,level,r"$C_\text{PortLLM}="+format_num(diff)+r"$",fontsize=26)
            else:
                mid=(loss_means[0]-loss_means[-1])/2+loss_means[-1]
                ax.text(0.1,mid,r"$B_\text{PortLLM}="+str(round(diff,2))+r"$",fontsize=26)
            if metric=='cost':
                epsilon=max([abs(vg)+ci for vg,ci in zip(vg_means,vg_95ci)])
                ax.plot(alphas,[epsilon]*len(alphas),color='gray',linestyle='dotted',linewidth=2)
                scale=0.7
                ax.text(0.1,scale*epsilon,r"$\epsilon="+str(round(epsilon,4))+r"$",fontsize=26)
            plt.tight_layout()
            postfix="_randn" if data.get('rand_dir',False) else ""
            fig.savefig(f"plots/{name}{postfix}.pdf")
            plt.close(fig)
            if data['t']==2: # only need once
                lgd_fig=plt.figure(figsize=(2,1.5),layout='constrained')
                h1,l1=ax.get_legend_handles_labels()
                h2,l2=ax2.get_legend_handles_labels()
                lgd_fig.legend([h1[0],h2[0],h1[1]],[l1[0],l2[0],l1[1]],fontsize=14)
                plt.subplots_adjust(left=0.2)
                try:
                    plt.savefig(f'plots/{metric}_legend{postfix}.pdf',bbox_inches='tight',format='pdf')
                except FileNotFoundError:
                    os.mkdir(r'plots')
                    plt.savefig(f'plots/{metric}_legend{postfix}.pdf',bbox_inches='tight',format='pdf')

def plot_vg_multireps(start_dir,dataset,metric, ts=[2,4,6,8,10],plot_postfix=""):
    # plot multiple repetitions on the same plot
    dset_names={'boolq':'BoolQ',
                'winogrande':'WinoGrande',
                'arc_easy':'ARC Easy',
                'arc_challenge':'ARC Challenge',
                'humaneval':'HumanEval',
                'gsm8k':'GSM8k'}
    markers={42:'o', # different marks for different random seeds for repetitions
             50:'v',
             75:'+'}
    for t in ts:
        fig,ax=plt.subplots(layout='constrained',figsize=(6.4,3.5))
        ax2=ax.twinx()
        for dir in os.listdir(start_dir):
            if not read_file(os.path.join(start_dir,dir),metric):
                continue
            data,name=read_file(os.path.join(start_dir,dir),metric)   
            if( not data['dataset']==dataset) or (not int(data['t'])==t):
                continue
            else:
                res=data['data']
                alphas=[]
                vg_means=[]
                vg_95ci=[]
                loss_means=[]
                loss_95ci=[]
                gn_means=[] # gn=grad norm
                gn_95ci=[]
                weights=data['token_lengths']
                for alpha,val in res.items():
                    alphas.append(float(alpha))
                    vg_mean,vg_std=weighted_stats(val['vg_samples'],weights)
                    vg_means.append(float(vg_mean))
                    vg_95ci.append(1.96*vg_std/(len(val['vg_samples'])**0.5))
                    loss_mean,loss_std=weighted_stats(val['loss_samples'],weights)
                    loss_means.append(float(loss_mean))
                    loss_95ci.append(1.96*loss_std/(len(val['loss_samples'])**0.5))
                    gn_mean,gn_std=weighted_stats(val['grad_norm_samples'],weights)
                    gn_means.append(gn_mean)
                    gn_95ci.append(1.96*gn_std/(len(val['grad_norm_samples'])**0.5))
                subscript="c" if metric=='cost' else 'g'
                norm_prod_label=r"$\|\nabla \ell_{\theta_t}\phi^{(\alpha)}\|\|\Delta_t^\text{(ft)}\|$" if metric=='cost' else r"$\|\nabla \ell_{\theta_t}((1-\alpha)\phi_0)\||\phi_0\|$"
                delta_ft_norm=data['delta_ft_norm']
                norms=[delta_ft_norm*gn for gn in gn_means]
                norm_95ci=[delta_ft_norm*ci for ci in gn_95ci]
                ax.errorbar(alphas,loss_means,loss_95ci,color='g',linestyle='dashed',label=r"$g_"+subscript+r"(\alpha)$",marker=markers.get(data['run'],""),markersize=6,linewidth=3)
                ax2.errorbar(alphas,norms,norm_95ci,color='b',linestyle='dotted',label=norm_prod_label,marker=markers.get(data['run'],""),markersize=6,linewidth=3)
                ax.errorbar(alphas,vg_means,vg_95ci,color='r',linestyle='solid',label=r"$\frac{d}{d\alpha} g_"+subscript+r"(\alpha)$",marker=markers.get(data['run'],""),markersize=6,linewidth=3)
                ax.set_xlabel(r"$\alpha$",fontsize=26)
                ax2.set_ylabel(r"Norm Product",fontsize=26,color='blue')
                ax.tick_params(axis='both',labelsize=20)
                ax2.tick_params(axis='both',labelsize=20)
                m_name=metric[0].upper()+ metric[1:]
                m_name=f"{dset_names[dataset]} t={data['t']} {m_name}"
                plt.title(f"t={t}",fontsize=26)
                if data.get('rand_dir',False):
                    plt.title(f"{dset_names[dataset]}",fontsize=26)
            plt.tight_layout()
            postfix="_randn" if data.get('rand_dir',False) else ""
            try:
                fig.savefig(f"plots/{data['dataset']}_{t}_{metric}{postfix}{plot_postfix}.pdf")
            except FileNotFoundError:
                os.mkdir(r"plots")
                fig.savefig(f"plots/{data['dataset']}_{t}_{metric}{postfix}{plot_postfix}.pdf")
            plt.close(fig)

start_dir=r'mistral_fw_1d_slice'

for bm in ['boolq','winogrande','arc_easy','arc_challenge']:
    plot_vg_multireps(start_dir,bm,'cost')
    plot_vg_multireps(start_dir,bm,'benefit')
    plot_vg(start_dir,bm,'cost')
    plot_vg(start_dir,bm,'benefit')

# plot random direction
for bm in ['boolq','winogrande','arc_easy','arc_challenge']:
    plot_vg_multireps(r'rand_t2',bm,'cost',ts=[2])

# 10 random directions for one repetition of boolq
plot_vg_multireps(r'bq_rand_10','boolq','cost',[2],"_10_dir")
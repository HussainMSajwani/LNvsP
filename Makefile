SHELL:=/bin/bash
#CEU hapmap3: https://mathgen.stats.ox.ac.uk/impute/data_download_1000G_pilot_plus_hapmap3.html
CEU_dir=/home/shussain/hapgen2/CEU_impute/

n_pheno=1
dir=.

sim: $(dir)/PS/output
#generate genotype in oxford format. chromosome $chr
$(dir)/prelim/prelim_$(chr):
	d_max=`wc -l < $(CEU_dir)hapmap3/hapmap3.r2.b36.allMinusPilot1CEU.chr$(chr).snpfilt.legend`; \
	if [ "$$d_max" -lt "$d" ]; then echo "Not enough SNPs in reference panel"; exit 64; fi; \
	rand=`awk -v d_max=$$d_max -v d=$$d 'BEGIN{srand();print int(rand()*(d_max-d))}'`; \
	first=`cat $(CEU_dir)/hapmap3/hapmap3.r2.b36.allMinusPilot1CEU.chr$(chr).snpfilt.legend | head -$$(($$rand+1)) | tail -1 | cut -d' ' -f2`; \
	dth=`cat $(CEU_dir)/hapmap3/hapmap3.r2.b36.allMinusPilot1CEU.chr$(chr).snpfilt.legend | head -$$(($$d+$$rand)) | tail -1 | cut -d' ' -f2`; \
	echo $$dth; \
	dummyDL=`sed -n '2'p $(CEU_dir)hapmap3/hapmap3.r2.b36.allMinusPilot1CEU.chr$(chr).snpfilt.legend | cut -d ' ' -f 2`; \
	~/hapgen2/hapgen2 -m $(CEU_dir)genetic_maps/genetic_map_chr$(chr)_combined_b36.txt\
					  -l $(CEU_dir)hapmap3/hapmap3.r2.b36.allMinusPilot1CEU.chr$(chr).snpfilt.legend\
					  -h $(CEU_dir)hapmap3/hapmap3.r2.b36.allMinusPilot1CEU.chr$(chr).snpfilt.haps\
					  -o $(dir)/prelim/prelim_$(chr) \
					  -n $(n) 0 \
					  -int $$first $$dth \
					  -dl $$first 0 0 0 \
					  -no_haps_output
	touch $@

#transform from oxford format into plink format
$(dir)/prelim/prelim_plink_chr$(chr): $(dir)/prelim/prelim_$(chr)
	~/plink/plink-1.9/plink --data $(dir)/prelim/prelim_$(chr).controls \
							--oxford-single-chr $(chr) \
							--make-bed \
							--out $(dir)/prelim/prelim_plink_chr$(chr)
	touch $@


$(dir)/PS/output: $(dir)/prelim/prelim_plink_chr$(chr)
	mkdir -p $(dir)/PS
	
	Rscript commandlineFunctions.R \
			--NrSamples=$(n) --NrPhenotypes=$(n_pheno) \
			--tNrSNP=$(d) --format="plink"\
			--genotypefile=$(dir)/prelim/prelim_plink_chr$(chr).bed \
			--cNrSNP=$(dc) \
			--genVar=0.95 --h2s=$(h2s) --phi=1 \
			--directory=$(dir)/PS \
			--subdirectory=output \
			--showProgress \
			--saveTable \
			--savePLINK \

	echo !!



move: 
	rm -rf temp
	rm -f $(dir)/prelim/*.gen 
	mkdir -p ~/Simulated_data/`date +'%d%m%Y'`/$(dir)
	cp $(dir) -r ~/Simulated_data/`date +'%d%m%Y'`/$(dir)

clean:
	rm -rf prelim
	rm -rf temp
	rm -rf final
	rm -rf PS
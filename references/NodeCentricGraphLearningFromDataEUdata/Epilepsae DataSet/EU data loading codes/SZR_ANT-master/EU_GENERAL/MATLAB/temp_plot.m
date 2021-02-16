%%
 % Code for checking results
                se_ftrs_z=zscore(se_ftrs')'; %z-score
                plv_ftrs_z=zscore(plv_ftrs')'; %z-score
                
                figure(1); clf();
                ax1=subplot(411);
                %imagesc(se_ftrs_z);
                h=plot(se_time_sec,se_ftrs_z);
                %title(sprintf('%s-%s, Szr%d',soz_chans_bi{cloop,1},soz_chans_bi{cloop,2},sloop));
                
                ax2=subplot(412); 
                %plot(se_time_sec,se_class,'--b'); hold on;
                plot(se_time_sec,se_szr_class,'r-');
                axis tight;
                
                ax3=subplot(413);
                plot(se_time_sec,plv_ftrs_z);
                axis tight;
                
                ax4=subplot(414);
                plot(targ_raw_ieeg_sec,targ_raw_ieeg);
                %plot(time_dec,ieeg);
                axis tight;
                
                linkaxes([ax1 ax2 ax3 ax4],'x');
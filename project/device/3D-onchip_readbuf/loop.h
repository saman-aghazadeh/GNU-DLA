
#define loop2(var1, limit1, var2, limit2) if(var1 == limit1 && var2 == limit2) \
		var1 = 0;	\
	else if(var2 == limit2)	\
		var1++;	\
	if(var2 == limit2)	\
		var2 = 0;	\
	else	\
		var2++;	\

#define loop3(var1, limit1, var2, limit2, var3, limit3) if(var1 == limit1 && var2 == limit2 && var3 == limit3) \
		var1 = 0;	\
	else if(var2 == limit2 && var3 == limit3)	\
		var1++;	\
	if(var2 == limit2 && var3 == limit3)	\
		var2 = 0;	\
	else if(var3 == limit3)	\
		var2++;	\
	if(var3 == limit3)	\
		var3 = 0;	\
	else	\
		var3++;

#define loop4(var1, limit1, var2, limit2, var3, limit3, var4, limit4) if(var1 == limit1 && var2 == limit2 && var3 == limit3 && var4 == limit4)	\
		var1 = 0;	\
	else if(var2 == limit2 && var3 == limit3 && var4 == limit4)	\
		var1++;	\
	if(var2 == limit2 && var3 == limit3 && var4 == limit4)	\
		var2 = 0;	\
	else if(var3 == limit3 && var4 == limit4)	\
		var2++;	\
	if(var3 == limit3 && var4 == limit4)	\
		var3 = 0;	\
	else if(var4 == limit4)	\
		var3++;	\
	if(var4 == limit4)	\
		var4 = 0;	\
	else	\
		var4++;

#define loop5(var1, limit1, var2, limit2, var3, limit3, var4, limit4, var5, limit5) if(var1 == limit1 && var2 == limit2 && var3 == limit3 && var4 == limit4 && var5 == limit5)	\
		var1 = 0;	\
	else if(var2 == limit2 && var3 == limit3 && var4 == limit4 && var5 == limit5)	\
		var1++;	\
	if(var2 == limit2 && var3 == limit3 && var4 == limit4 && var5 == limit5)	\
		var2 = 0;	\
	else if(var3 == limit3 && var4 == limit4 && var5 == limit5)	\
		var2++;	\
	if(var3 == limit3 && var4 == limit4 && var5 == limit5)	\
		var3 = 0;	\
	else if(var4 == limit4 && var5 == limit5)	\
		var3++;	\
	if(var4 == limit4 && var5 == limit5)	\
		var4 = 0;	\
	else if(var5 == limit5)	\
		var4++;	\
	if(var5 == limit5) \
		var5 = 0;	\
	else \
		var5++;


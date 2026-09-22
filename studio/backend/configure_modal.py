"""Transfer explicitly named GitHub secrets to Modal without printing their values."""
import os
import modal
required=['STUDIO_TOKEN']
for name in required:
    if len(os.getenv(name,''))<32:raise SystemExit(name+' must contain at least 32 characters.')
names=['STUDIO_TOKEN','OMNIROUTE_API_KEY','OMNIROUTE_BASE_URL','OMNIROUTE_MODEL','FREELLMAPI_API_KEY','FREELLMAPI_BASE_URL','FREELLMAPI_MODEL','MVSEP_API_KEY','MVSEP_SEP_TYPE']
values={name:os.getenv(name,'') for name in names}
if not any(all(os.getenv(prefix+'_'+field) for field in ['API_KEY','BASE_URL','MODEL']) for prefix in ['OMNIROUTE','FREELLMAPI']):
    raise SystemExit('Configure API_KEY, BASE_URL and MODEL for at least one translation provider.')
modal.Secret.objects.create('videotrans-studio',values,allow_existing=True)
modal.Secret.from_name('videotrans-studio').update(values)
print('Modal service secrets configured.')

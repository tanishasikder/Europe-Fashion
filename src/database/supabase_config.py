from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

supabase: Client = create_client(
    os.environ.get('SUPABASE_URL'), 
    os.environ.get('SUPABASE_KEY')
)

# Insert a new row into table
new_row = {'color' : 'red', 'category' : 'dress', 'size': 'L', 'price': 34.99}
supabase.table('clothes').insert(new_row).execute()

newer_row = {'color' : 'blue', 'category' : 'dress', 'size': 'L', 'price': 34.99}
# executes whichever is the second id
supabase.table('clothes').update(newer_row).eq('id', 2).execute()

supabase.table('clothes').delete().eq('id', 2).execute()

supabase.table('clothes').select().eq('id', 2).execute()

# Get an image stored in supabase
response = supabase.storage.from_('clothes').get_public_url('dress.jpg')
results = supabase.table('clothes').select('*').execute()



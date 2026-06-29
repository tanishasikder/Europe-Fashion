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

results = supabase.table('clothes').select('*').execute()
print(results)


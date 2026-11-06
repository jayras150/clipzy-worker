-- Clipzy AI: initial Supabase schema
-- Apply this file in Supabase Dashboard > SQL Editor.
-- Safe to re-run: tables, indexes, policies, triggers and publication setup are idempotent.

create extension if not exists pgcrypto;

-- Keep profile rows in sync with Supabase Auth users.
create table if not exists public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    display_name text,
    avatar_url text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.user_settings (
    user_id uuid primary key references auth.users(id) on delete cascade,
    default_subtitle_style text not null default 'VIRAL'
        check (default_subtitle_style in (
            'NONE','VIRAL','PODCAST','IMPACT','NEON','MAGENTA','GOLD','CRIMSON',
            'CYBER','BEAST','MARKER','RETRO','GLITCH','MINIMAL','EMERALD','SUNSET','VIOLET'
        )),
    default_subtitle_language text not null default '',
    default_aspect_ratio text not null default '9:16'
        check (default_aspect_ratio in ('source','9:16','16:9','1:1','4:5','4:3','3:4','21:9')),
    default_output_quality text not null default 'original'
        check (default_output_quality in ('original','1080p','720p','480p')),
    preserve_source_quality boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- job_id is the identifier sent to the Python worker. UUID text is accepted by
-- the worker's [A-Za-z0-9._-]{1,100} validation rule.
create table if not exists public.jobs (
    id uuid primary key default gen_random_uuid(),
    job_id text not null unique,
    user_id uuid not null references auth.users(id) on delete cascade,

    source_url text not null,
    status text not null default 'pending'
        check (status in ('pending','queued','running','completed','failed','cancelled')),
    progress integer not null default 0 check (progress between 0 and 100),
    current_stage text,
    stage_label text,

    -- Store the exact validated payload sent to the worker. Never store secrets here.
    request_payload jsonb not null default '{}'::jsonb,

    -- Vast.ai tracking fields; session_id may be null until a session is opened.
    vast_endpoint_id text,
    vast_session_id text,
    worker_job_id text,

    error_message text,
    worker_response jsonb,
    created_at timestamptz not null default now(),
    started_at timestamptz,
    finished_at timestamptz,
    updated_at timestamptz not null default now()
);

create table if not exists public.job_events (
    id bigint generated always as identity primary key,
    job_id text not null references public.jobs(job_id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    event_type text not null,
    status text,
    progress integer check (progress between 0 and 100),
    stage text,
    message text,
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.job_outputs (
    id uuid primary key default gen_random_uuid(),
    job_id text not null references public.jobs(job_id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    output_type text not null check (output_type in ('primary_video','clip','manifest','thumbnail','other')),
    clip_index integer,
    object_key text,
    public_url text,
    file_size_bytes bigint,
    content_type text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_jobs_user_created
    on public.jobs(user_id, created_at desc);
create index if not exists idx_jobs_status_created
    on public.jobs(status, created_at desc);
create index if not exists idx_job_events_job_created
    on public.job_events(job_id, created_at asc);
create index if not exists idx_job_outputs_job
    on public.job_outputs(job_id);

-- updated_at trigger helper.
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_profiles_updated_at on public.profiles;
create trigger trg_profiles_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

drop trigger if exists trg_user_settings_updated_at on public.user_settings;
create trigger trg_user_settings_updated_at
before update on public.user_settings
for each row execute function public.set_updated_at();

drop trigger if exists trg_jobs_updated_at on public.jobs;
create trigger trg_jobs_updated_at
before update on public.jobs
for each row execute function public.set_updated_at();

-- Automatically create a profile and settings row after sign-up.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    insert into public.profiles (id, display_name, avatar_url)
    values (
        new.id,
        coalesce(new.raw_user_meta_data ->> 'name', new.raw_user_meta_data ->> 'full_name'),
        new.raw_user_meta_data ->> 'avatar_url'
    )
    on conflict (id) do nothing;

    insert into public.user_settings (user_id)
    values (new.id)
    on conflict (user_id) do nothing;

    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute procedure public.handle_new_user();

-- Enable RLS on every user-facing table.
alter table public.profiles enable row level security;
alter table public.user_settings enable row level security;
alter table public.jobs enable row level security;
alter table public.job_events enable row level security;
alter table public.job_outputs enable row level security;

-- Profiles: users can read/update only their own profile.
drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own" on public.profiles
for select to authenticated using (id = (select auth.uid()));

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own" on public.profiles
for update to authenticated using (id = (select auth.uid()))
with check (id = (select auth.uid()));

-- Settings: users can read/create/update only their own settings.
drop policy if exists "settings_select_own" on public.user_settings;
create policy "settings_select_own" on public.user_settings
for select to authenticated using (user_id = (select auth.uid()));

drop policy if exists "settings_insert_own" on public.user_settings;
create policy "settings_insert_own" on public.user_settings
for insert to authenticated with check (user_id = (select auth.uid()));

drop policy if exists "settings_update_own" on public.user_settings;
create policy "settings_update_own" on public.user_settings
for update to authenticated using (user_id = (select auth.uid()))
with check (user_id = (select auth.uid()));

-- Jobs/events/outputs are read-only to app clients.
-- Writes are performed by trusted Edge Functions using the service role key.
drop policy if exists "jobs_select_own" on public.jobs;
create policy "jobs_select_own" on public.jobs
for select to authenticated using (user_id = (select auth.uid()));

drop policy if exists "job_events_select_own" on public.job_events;
create policy "job_events_select_own" on public.job_events
for select to authenticated using (user_id = (select auth.uid()));

drop policy if exists "job_outputs_select_own" on public.job_outputs;
create policy "job_outputs_select_own" on public.job_outputs
for select to authenticated using (user_id = (select auth.uid()));

-- Realtime: allow clients to subscribe to changes in their jobs table.
do $$
begin
    if not exists (
        select 1
        from pg_publication_tables
        where pubname = 'supabase_realtime'
          and schemaname = 'public'
          and tablename = 'jobs'
    ) then
        alter publication supabase_realtime add table public.jobs;
    end if;
end;
$$;

comment on table public.jobs is
'Clipzy processing jobs. Client writes must go through authenticated Edge Functions.';
comment on column public.jobs.request_payload is
'Validated worker request payload. Do not put API keys or credentials in this JSON.';
comment on table public.job_outputs is
'Cloudflare R2 output object metadata; video bytes are stored in R2, not PostgreSQL.';

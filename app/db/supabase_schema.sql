-- Create schema for disclosure information
CREATE TABLE IF NOT EXISTS disclosure_info (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_name TEXT NOT NULL,
    disclosure_date DATE NOT NULL,
    disclosure_type TEXT NOT NULL,
    title TEXT NOT NULL,
    file_url TEXT,
    file_name TEXT,
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_disclosure_date ON disclosure_info(disclosure_date);
CREATE INDEX IF NOT EXISTS idx_company_name ON disclosure_info(company_name);
CREATE INDEX IF NOT EXISTS idx_disclosure_type ON disclosure_info(disclosure_type);

-- Create trigger function to update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger
CREATE TRIGGER update_disclosure_info_updated_at
    BEFORE UPDATE ON disclosure_info
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

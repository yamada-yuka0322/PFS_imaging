#!/usr/bin/env python
# Based on
# https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/blob/master/dr3/catalogQuery/hscSspQuery3.py

import os
import json
import time
import getpass
import argparse
import urllib.request, urllib.error, urllib.parse
import astropy.io.ascii as ascii
import astropy.io.fits as pyfits
import yaml
from pathlib import Path

version =   20190924.1
diffver =   '-colorterm'
release_version =   'dr4-citus'
doDownload = True
doUnzip = True

"""
This script downloads HSC catalog data from the HSC SSP CAS API.

Workflow:
1. Read a YAML config file to locate the SQL template and output directory.
2. Read the target tract list from Tracttest.csv.
3. Replace the {$tract} placeholder in the SQL file with the tract IDs.
4. Submit the SQL query as a catalog job to the HSC CAS server.
5. Wait until the job finishes.
6. Download the result as one file per tract group.
7. Split the downloaded FITS table into separate FITS files for each tract.

Expected SQL template:
    The SQL file must contain the placeholder {$tract}, which will be replaced
    by a comma-separated list of tract IDs.

Output structure:
    output_dir/kind/tract_group/
        Downloaded files for each tract group.
    output_dir/kind/tract/
        FITS files separated by individual tract.
"""

def chunkNList(seq, num):
    """
    fuction to divide the tracts into num groups.
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def GetSQLPath(kind, config):
    if (kind=='star'):
        path = "../sql/star_default.sql"
    elif (kind=='patchqa'):
        path = "../sql/s23b_wide_patches.sql"
    elif (kind == 'random'):
        path = os.path.join(config["target_selection"]["pfstarget"], config["target_selection"]["random_sql"])
    elif (kind == 'galaxy'):
        path = os.path.join(config["target_selection"]["pfstarget"], config["target_selection"]["galaxy_sql"])
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return path

def GetNgroups(kind):
    """
    number of tract group. All the tracts in the same tract group would be downloaded at once.
    """
    if (kind == 'patchqa'):
        return 1
    else:
        return 40

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--user', '-u', required=True,
                        help='specify your STARS account')
    parser.add_argument('--delete-job', '-D', action='store_true',
                        help='delete job after your downloading')
    parser.add_argument('--format', '-f', dest='out_format', default='fits',
                        choices=['csv', 'csv.gz', 'sqlite3', 'fits'],
                        help='specify output format')
    parser.add_argument('--nomail', '-M', action='store_true',
                        help='suppress email notice')
    parser.add_argument('--password-env', default='HSC_SSP_CAS_PASSWORD',
                        help='environment variable for STARS')
    parser.add_argument('--api-url',
                        default='https://hscdata.mtk.nao.ac.jp/datasearch/api/catalog_jobs/',
                        help='for developers')
    parser.add_argument('--skip-syntax-check', '-S', action='store_true',
                        help='skip syntax check')
    parser.add_argument("--config", "-c", default=None,
                        help="YAML config file containing paths and SQL settings")
    parser.add_argument("--kind", default=None,
                        choices=["star", "patchqa", "galaxy", "random"],
                        help="which catalog to download")

    args = parser.parse_args()
    release_year   =   's23'

    config = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    if args.kind is not None and config:
        sql_file = Path(GetSQLPath(args.kind, config))
        output_dir = Path(config["data"][f"{args.kind}_dir"]).expanduser()
    else:
        parser.error("--kind and --config are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # tracts for DR4 S23B
    ngroups =   GetNgroups(args.kind)
    tractname=  config['tractlist']
    
    # randoms or objects 
    prefix  =   f"{output_dir}/tract_group" #directory to save the downloaded tract groups
    prefix2 =   f"{output_dir}/tract" #directory to save the objects separated per each tract.
    if not os.path.exists(prefix2):
        os.system('mkdir -p %s' %prefix2)
        
    if not os.path.exists(prefix):
        os.system('mkdir -p %s' %prefix)
    
    sql         =   sql_file.read_text()
    tracts      =   ascii.read(tractname)['tract']
    tracts2     =   chunkNList(tracts,ngroups)
    if doDownload:
        credential  =   {'account_name': args.user, 'password': getPassword(args)}
        for ig,tractL in enumerate(tracts2):
            print('Group: %s' %ig)
            downloadTracts(ig,tractL, prefix, credential, sql, args)
            if doUnzip:
                print('unzipping group: %s' %ig)
                separateTracts(ig,tractL, prefix, prefix2, args)
    
    return

def separateTracts(ig,tractL, prefix, prefix2, args):
    """
    function to separate the downloaded objects into tracts.
    """
    if args.out_format != "fits":
        raise ValueError("separateTracts currently supports only fits format")
    
    infname     =   '%s.%s'%(ig,args.out_format)
    infname     =   os.path.join(prefix,infname)
    if not os.path.exists(infname):
        print('Does not have input file')
        return
    fitsAll     =   pyfits.getdata(infname)
    print('read %s galaxies' %len(fitsAll))

    for tract in tractL:
        outfname    =   os.path.join(prefix2,'%s.fits' %(tract))
        if os.path.exists(outfname):
            print('already have file for tract: %s' \
                    %tract)
            continue
        fits        =   fitsAll[fitsAll['tract']==int(tract)]
        if len(fits)>10:
            pyfits.writeto(outfname,fits)
        del fits
    return

def downloadTracts(ig, tractL, prefix, credential, sql, args):
    """
    Submit a query to the HSC CAS server and download the catalog.

    The placeholder {$tract} in the SQL template is replaced by the
    tract IDs contained in tractL.

    Parameters
    ----------
    ig : int
        Tract group index.

    tractL : list
        List of tract IDs included in the group.

    prefix : str
        Output directory.

    credential : dict
        User account and password.

    sql : str
        SQL template.

    args : argparse.Namespace

    Outputs
    -------
    prefix/ig.fits
    """
    tractStr    =   map(str,tractL)
    tname       =   "'{0}'".format("', '".join(tractStr))
    job         =   None
    
    # Replace {$tract} in the SQL template with the tract list
    sqlU        =   sql.replace('{$tract}',tname)
    outfname = '%s.%s'%(ig,args.out_format)
    outfname    =   os.path.join(prefix,outfname)
    if os.path.exists(outfname):
        print('already have output')
        return
    print(f"downloading tracts: {tractL}")
    print('querying data')
    job         =   submitJob(credential, sqlU, args)
    blockUntilJobFinishes(credential, job['id'], args)
    print('downloading data')
    with open(outfname, "wb") as fileOut:
        download(credential, job["id"], fileOut, args)
    if args.delete_job:
        deleteJob(credential, job['id'], args)
    print('closing output file')
    return

class QueryError(Exception):
    pass

def httpJsonPost(url, data):
    data['clientVersion'] = version
    postData = json.dumps(data)
    return httpPost(url, postData, {'Content-type': 'application/json'})

def httpPost(url, postData, headers):
    req = urllib.request.Request(url, postData.encode('utf-8'), headers)
    res = urllib.request.urlopen(req)
    return res

def submitJob(credential, sql, args):
    url = args.api_url + 'submit'
    catalog_job = {
        'sql'                     : sql,
        'out_format'              : args.out_format,
        'include_metainfo_to_body': True,
        'release_version'         : release_version,
    }
    postData = {'credential': credential, 'catalog_job': catalog_job, 'nomail': args.nomail, 'skip_syntax_check': args.skip_syntax_check}
    res = httpJsonPost(url, postData)
    job = json.load(res)
    return job

def jobStatus(credential, job_id, args):
    url = args.api_url + 'status'
    postData = {'credential': credential, 'id': job_id}
    res = httpJsonPost(url, postData)
    job = json.load(res)
    return job


def blockUntilJobFinishes(credential, job_id, args):
    """
    Wait until the submitted HSC CAS job finishes.

    This function periodically checks the job status using the HSC CAS API.
    excessive requests to the server, with a maximum interval of 30 seconds.

    Parameters
    ----------
    credential : dict
        User account information containing account name and password.

    job_id : int
        Job ID returned by submitJob().

    args : argparse.Namespace
        Command line arguments.

    Raises
    ------
    QueryError
        Raised if the job status becomes 'error'.

    Returns
    -------
    None
        Returns after the job status becomes 'done'.
    """

    max_interval = 0.5 * 60 # sec.
    interval = 1
    while True:
        time.sleep(interval)
        job = jobStatus(credential, job_id, args)
        if job['status'] == 'error':
            raise QueryError('query error: ' + job['error'])
        if job['status'] == 'done':
            break
        interval *= 2
        if interval > max_interval:
            interval = max_interval
    print('blocking over')
    return

def download(credential, job_id, out, args):
    url     =   args.api_url + 'download'
    postData=   {'credential': credential, 'id': job_id}
    res     =   httpJsonPost(url, postData)
    bufSize =   1024 * 1<<10 # 1024K
    bufLim  =   1 * 1<<10 # 1K
    while True:
        buf = res.read(bufSize)
        out.write(buf)
        if len(buf) < bufLim:
            break
    return

def deleteJob(credential, job_id, args):
    url = args.api_url + 'delete'
    postData = {'credential': credential, 'id': job_id}
    httpJsonPost(url, postData)
    return

def getPassword(args):
    password_from_envvar = os.environ.get(args.password_env, '')
    if password_from_envvar != '':
        return password_from_envvar
    else:
        return getpass.getpass('password? ')

if __name__ == '__main__':
    main()

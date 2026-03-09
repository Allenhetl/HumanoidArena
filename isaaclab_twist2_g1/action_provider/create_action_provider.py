from action_provider.action_provider_dds import DDSActionProvider
from action_provider.action_provider_replay import FileActionProviderReplay
from action_provider.action_provider_wh_twist2 import DDSRLActionProvider
from action_provider.action_provider_openpi import OpenPIActionProvider
from pathlib import Path


def create_action_provider(env,args):
    """create action provider based on parameters"""
    if args.action_source == "dds":
        return DDSActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "dds_wholebody":
        return DDSRLActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "sonic_wholebody":
        from action_provider.action_provider_sonic import SonicActionProvider
        return SonicActionProvider(env=env, args_cli=args)
    elif args.action_source == "openpi":
        return OpenPIActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "replay":
        return FileActionProviderReplay(env=env,args_cli=args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None

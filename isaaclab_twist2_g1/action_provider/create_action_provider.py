from action_provider.action_provider_replay import FileActionProviderReplay
from action_provider.action_provider_wh_twist2 import TWIST2ActionProvider
from action_provider.action_provider_openpi import OpenPIActionProvider


def create_action_provider(env,args):
    """create action provider based on parameters"""
    if args.action_source == "twist2_wholebody":
        return TWIST2ActionProvider(
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
        if args.gmt_backend == "twist2":
            return TWIST2ActionProvider(env=env, args_cli=args)
        elif args.gmt_backend == "sonic":
            from action_provider.action_provider_sonic import SonicActionProvider
            return SonicActionProvider(env=env, args_cli=args)
        return FileActionProviderReplay(env=env,args_cli=args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None

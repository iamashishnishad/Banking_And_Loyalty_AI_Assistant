"""Mint a JWT for local testing (stands in for a real identity provider). Prints only the token.

    docker compose exec api python -m scripts.issue_token --tenant demo-bank --role customer --customer CUST1001
    docker compose exec api python -m scripts.issue_token --tenant demo-bank --role admin
"""

import argparse

from app.services.auth import issue_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", required=True, help="Tenant slug, e.g. demo-bank")
    parser.add_argument("--role", choices=["customer", "staff", "admin"], default="customer")
    parser.add_argument("--customer", help="customer_ref; required for --role customer")
    parser.add_argument("--subject", help="Token subject; defaults to customer:<ref> or <role>:<random>")
    parser.add_argument("--ttl", type=int, help="Lifetime in seconds (default JWT_TTL_SECONDS)")
    args = parser.parse_args()
    if args.role == "customer" and not args.customer:
        parser.error("--customer is required for --role customer")
    print(issue_token(tenant=args.tenant, role=args.role, customer_ref=args.customer, subject=args.subject, ttl_seconds=args.ttl))


if __name__ == "__main__":
    main()

